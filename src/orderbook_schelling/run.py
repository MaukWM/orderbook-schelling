"""Run an experiment from a TOML config: every trial's seed, models and settings come from the config file.

usage: uv run python -m orderbook_schelling.run experiments/<name>.toml
Results append to runs/<name>.jsonl, one row per finished trial. A rerun skips finished trials and retries trials
that hit an API error (those are never written). The code and config must be committed unless allow_dirty = true.
"""

import hashlib
import json
import subprocess
import sys
import threading
import time
import tomllib
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any

from orderbook_schelling.backends import Reply, Route, ask
from orderbook_schelling.feed import build_feed, clean_feed, recent_orders, render_tape
from orderbook_schelling.parse import parse_receiver, parse_sender
from orderbook_schelling.prompts import receiver_prompt, sender_prompt

REPO = Path(__file__).resolve().parents[2]
CONDITIONS = ("anon", "monitored")
TRACKED = ("src", "experiments", "pyproject.toml", "uv.lock")
OPERATIONAL = ("budget_usd", "max_usd_per_trial", "workers", "allow_dirty")  # can change between runs of one experiment


@dataclass(frozen=True)
class Model:
    route: Route
    id: str
    usd_per_m_in: float
    usd_per_m_out: float
    params: dict[str, Any]


def experiment_hash(cfg: dict[str, Any]) -> str:
    """Hash of every config key that defines the experiment; operational keys are left out so a run can resume."""
    defining = {k: v for k, v in cfg.items() if k not in OPERATIONAL}
    return hashlib.sha256(json.dumps(defining, sort_keys=True).encode()).hexdigest()[:16]


def git_state() -> tuple[str, bool]:
    """The commit the code ran at and whether code, configs or dependencies had uncommitted changes."""
    git = ["git", "-C", str(REPO)]
    log = [*git, "log", "-1", "--format=%H", "--", *TRACKED]  # last commit touching code or configs, not results
    commit = subprocess.run(log, capture_output=True, text=True, check=True).stdout.strip()
    status = subprocess.run([*git, "status", "--porcelain", "--", *TRACKED], capture_output=True, text=True).stdout
    return commit, bool(status.strip())


class Experiment:
    """A loaded config plus the bookkeeping shared by its trials: output file, spend and the budget cap."""

    def __init__(self, path: Path) -> None:
        self.cfg: dict[str, Any] = tomllib.loads(path.read_text())
        commit, dirty = git_state()
        if dirty and not self.cfg.get("allow_dirty"):
            sys.exit("uncommitted changes in src/, experiments/ or dependencies; commit first (or allow_dirty = true)")
        self.meta = {
            "config": path.name,
            "config_sha256": experiment_hash(self.cfg),
            "code": commit + ("-dirty" if dirty else ""),
        }
        self.models = {name: self.load_model(name, m) for name, m in self.cfg["models"].items()}
        names = {n for pair in self.cfg["pairs"] for n in pair} | set(self.cfg.get("clean_receivers", []))
        if missing := names - self.models.keys():
            sys.exit(f"pairs or clean_receivers name unknown models: {sorted(missing)}")
        self.out = REPO / "runs" / f"{self.cfg['name']}.jsonl"
        self.out.parent.mkdir(exist_ok=True)
        rows = self.read_rows()
        own = (self.meta["config_sha256"], self.meta["code"])
        if foreign := {(r["config_sha256"], r["code"]) for r in rows} - {own}:
            sys.exit(
                f"{self.out} holds rows from another config or commit {sorted(foreign)}; rename the file or the run"
            )
        self.done = {row["trial"] for row in rows}
        self.spent = sum(row["cost_usd"] for row in rows)
        self.in_flight = 0
        self.lock = threading.Lock()

    def load_model(self, name: str, m: dict[str, Any]) -> Model:
        """A model entry; route_all = "openrouter" sends every model through OpenRouter under its openrouter_id."""
        if self.cfg.get("route_all") == "openrouter":
            return Model(
                "openrouter", m["openrouter_id"], m["usd_per_m_in"], m["usd_per_m_out"], m.get("openrouter_params", {})
            )
        return Model(m["route"], m["id"], m["usd_per_m_in"], m["usd_per_m_out"], m.get("params", {}))

    def read_rows(self) -> list[dict[str, Any]]:
        """Finished rows; a half-written last line from an interrupted run is ignored."""
        if not self.out.exists():
            return []
        lines = self.out.read_text().splitlines()
        rows = [json.loads(line) for line in lines[:-1]]
        try:
            rows.append(json.loads(lines[-1]))
        except (IndexError, json.JSONDecodeError):
            pass
        return rows

    def start(self, trial: str) -> bool:
        """Claim a trial unless it is done or the budget, counting trials still running at their cap, is used up."""
        with self.lock:
            reserved = self.in_flight * self.cfg["max_usd_per_trial"]
            if trial in self.done or self.spent + reserved >= self.cfg["budget_usd"]:
                return False
            self.done.add(trial)
            self.in_flight += 1
            return True

    def finish(self, trial: str, row: dict[str, Any] | None, spent_on_failure: float = 0.0) -> None:
        """Record a finished trial; a trial with an API error is released for a retry, but its cost still counts."""
        with self.lock:
            self.in_flight -= 1
            if row is None:
                self.spent += spent_on_failure
                self.done.discard(trial)
                return
            self.spent += row["cost_usd"]
            with self.out.open("a") as f:
                f.write(json.dumps({**self.meta, "trial": trial, "time": time.time(), **row}) + "\n")

    def call(self, name: str, prompt: str) -> tuple[Reply, float]:
        """One model call and its cost at the config's per-million-token prices."""
        m = self.models[name]
        reply = ask(m.route, m.id, prompt, m.params)
        return reply, (reply.input_tokens * m.usd_per_m_in + reply.output_tokens * m.usd_per_m_out) / 1e6

    def request(self, name: str) -> dict[str, Any]:
        """What was asked of a model: route, id and request parameters."""
        m = self.models[name]
        return {"route": m.route, "model_id": m.id, "params": m.params}


def sender_status(reply: Reply, parsed_status: str) -> str:
    """Parser status (sent/empty/unparsed), or refused on a provider refusal/filter stop, or truncated at the cap."""
    if reply.stop_reason in ("refusal", "content_filter"):
        return "refused"
    if parsed_status == "unparsed" and reply.stop_reason in ("max_tokens", "length"):
        return "truncated"
    return parsed_status


def message_trial(exp: Experiment, sender: str, receiver: str, condition: str, word: str, rep: int) -> None:
    """Sender sees the recent feed and places orders; if any are valid, the receiver reads the tape and guesses."""
    trial = f"{exp.cfg['base_seed']}|{sender}|{receiver}|{condition}|{word}|{rep}"
    if not exp.start(trial):
        return
    market = f"{exp.cfg['base_seed']}|{word}|{rep}"  # every pair and condition sees the same market for a word and rep
    prompt = sender_prompt(word, recent_orders(market), condition == "monitored")
    s_reply, cost = exp.call(sender, prompt)
    if s_reply.error:
        print(f"retry later: {trial}: {s_reply.error}", file=sys.stderr)
        return exp.finish(trial, None, cost)
    parsed = parse_sender(s_reply.text)
    row: dict[str, Any] = {
        "kind": "message",
        "sender": sender,
        "receiver": receiver,
        "condition": condition,
        "word": word,
        "rep": rep,
        "sender_request": exp.request(sender),
        "sender_prompt": prompt,
        "sender_reply": asdict(s_reply),
        "status": sender_status(s_reply, parsed.status),
        "orders": [asdict(o) for o in parsed.orders],
        "rejected": [asdict(r) for r in parsed.rejected],
        "decoded": False,
    }
    if row["status"] == "sent":
        feed = build_feed(parsed.orders, market)
        tape = render_tape(feed)
        r_reply, r_cost = exp.call(receiver, receiver_prompt(tape))
        if r_reply.error:
            print(f"retry later: {trial}: {r_reply.error}", file=sys.stderr)
            return exp.finish(trial, None, cost + r_cost)
        cost += r_cost
        guess = parse_receiver(r_reply.text)
        row |= {
            "receiver_request": exp.request(receiver),
            "tape": tape,
            "from_sender": [e.from_sender for e in feed],
            "receiver_reply": asdict(r_reply),
            "guess": guess,
            "decoded": guess == word,
        }
    exp.finish(trial, row | {"cost_usd": cost})


def clean_trial(exp: Experiment, receiver: str, word: str, rep: int) -> None:
    """Negative control: the receiver reads a tape with no message; naming this trial's word is a false decode."""
    trial = f"{exp.cfg['base_seed']}|clean|{receiver}|{word}|{rep}"
    if not exp.start(trial):
        return
    tape = render_tape(clean_feed(f"{exp.cfg['base_seed']}|{word}|{rep}"))
    r_reply, cost = exp.call(receiver, receiver_prompt(tape))
    if r_reply.error:
        print(f"retry later: {trial}: {r_reply.error}", file=sys.stderr)
        return exp.finish(trial, None, cost)
    guess = parse_receiver(r_reply.text)
    exp.finish(
        trial,
        {
            "kind": "clean",
            "receiver": receiver,
            "word": word,
            "rep": rep,
            "receiver_request": exp.request(receiver),
            "tape": tape,
            "receiver_reply": asdict(r_reply),
            "guess": guess,
            "decoded": guess == word,
            "guess_in_words": guess in exp.cfg["words"],
            "cost_usd": cost,
        },
    )


def jobs(exp: Experiment) -> list[Callable[[], None]]:
    """Every trial once, ordered rep by rep and word by word so a budget stop leaves a balanced design."""
    cfg, out = exp.cfg, list[Callable[[], None]]()
    pairs = list(dict.fromkeys(tuple(p) for p in cfg["pairs"]))
    for rep in range(cfg["reps"]):
        for word in dict.fromkeys(cfg["words"]):
            out += [partial(message_trial, exp, s, r, c, word, rep) for s, r in pairs for c in CONDITIONS]
            out += [partial(clean_trial, exp, r, word, rep) for r in dict.fromkeys(cfg.get("clean_receivers", []))]
    return out


def main(path: str) -> None:
    exp = Experiment(Path(path))
    with ThreadPoolExecutor(exp.cfg["workers"]) as pool:
        list(pool.map(lambda job: job(), jobs(exp)))
    budget = exp.cfg["budget_usd"]
    print(f"{exp.out}: spent ${exp.spent:.2f} of ${budget}" + (" (budget reached)" if exp.spent >= budget else ""))


if __name__ == "__main__":
    main(sys.argv[1])
