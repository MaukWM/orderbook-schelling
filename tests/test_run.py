import json
import re
from pathlib import Path
from typing import Any

import pytest

from orderbook_schelling import run
from orderbook_schelling.backends import Reply

CONFIG = """
name = "t"
base_seed = "s"
words = ["HELLO", "TIGER"]
reps = 1
budget_usd = {budget}
max_usd_per_trial = 0.01
workers = 1
pairs = [["a", "a"], ["refuser", "a"], ["a", "a"]]
clean_receivers = ["a"]

[models.a]
route = "openai"
id = "a-1"
usd_per_m_in = 1.0
usd_per_m_out = 1.0

[models.refuser]
route = "anthropic"
id = "r-1"
usd_per_m_in = 1.0
usd_per_m_out = 1.0
"""


CODEBOOK = {"HELLO": 311, "TIGER": 487}


class FakeModels:
    """Sender and receiver sharing an arbitrary codebook (word to one order size); can fail or refuse on demand."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.fail_receivers = False

    def __call__(self, route: str, model: str, prompt: str, params: dict[str, Any]) -> Reply:
        self.calls.append(model)
        if model == "r-1":
            return Reply("", "refusal", model, route, 100, 0)
        if word := re.search(r"Your message is the word: (\w+)\.", prompt):
            orders = [{"side": "buy", "price": 137.0, "qty": CODEBOOK[word.group(1)]}]
            return Reply(json.dumps({"orders": orders}), "stop", model, route, 100, 100)
        if self.fail_receivers:
            return Reply("", None, model, route, 0, 0, error="APIError: down")
        qtys = {int(q) for q in re.findall(r"buy\s+137\.00\s+(\d+)", prompt)}
        guess = next((w for w, q in CODEBOOK.items() if q in qtys), "DAY")
        return Reply(json.dumps({"word": guess}), "stop", model, route, 100, 100)


@pytest.fixture
def setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[FakeModels, Path]:
    fake = FakeModels()
    monkeypatch.setattr(run, "ask", fake)
    monkeypatch.setattr(run, "REPO", tmp_path)
    monkeypatch.setattr(run, "git_state", lambda: ("abc", False))
    return fake, tmp_path


def config(tmp_path: Path, budget: float = 10.0) -> Path:
    path = tmp_path / "t.toml"
    path.write_text(CONFIG.format(budget=budget))
    return path


def rows(tmp_path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (tmp_path / "runs" / "t.jsonl").read_text().splitlines()]


def test_full_run_records_each_trial_once_and_decodes(setup: tuple[FakeModels, Path]) -> None:
    fake, tmp = setup
    run.main(str(config(tmp)))
    out = rows(tmp)
    assert len(out) == 2 * 2 * 2 + 2  # 2 unique pairs x 2 conditions x 2 words + 2 clean
    sent = [r for r in out if r["kind"] == "message" and r["sender"] == "a"]
    assert all(r["status"] == "sent" and r["decoded"] for r in sent)
    refused = [r for r in out if r["kind"] == "message" and r["sender"] == "refuser"]
    assert all(r["status"] == "refused" and r["decoded"] is False and "tape" not in r for r in refused)
    assert all(r["kind"] != "clean" or r["decoded"] is False for r in out)
    tapes = {r["condition"]: r["tape"] for r in sent if r["word"] == "HELLO"}
    assert tapes["anon"] == tapes["monitored"]  # same market for both conditions
    calls = len(fake.calls)
    run.main(str(config(tmp)))
    assert len(fake.calls) == calls and len(rows(tmp)) == len(out)  # rerun skips finished trials


def test_api_errors_are_not_recorded_and_are_retried(setup: tuple[FakeModels, Path]) -> None:
    fake, tmp = setup
    fake.fail_receivers = True
    run.main(str(config(tmp)))
    assert not any(r.get("receiver_reply") for r in rows(tmp))
    fake.fail_receivers = False
    run.main(str(config(tmp)))
    assert sum(r["kind"] == "message" and r["decoded"] for r in rows(tmp)) == 4


def test_budget_stops_new_trials_once_a_trial_at_its_cap_would_not_fit(setup: tuple[FakeModels, Path]) -> None:
    _, tmp = setup
    run.main(str(config(tmp, budget=0.0105)))  # cap 0.01 per trial, a fake trial costs 0.0004
    assert len(rows(tmp)) == 2 and sum(r["cost_usd"] for r in rows(tmp)) <= 0.0105


def test_raising_the_budget_resumes_but_another_experiment_refuses_to_mix(setup: tuple[FakeModels, Path]) -> None:
    _, tmp = setup
    run.main(str(config(tmp, budget=0.0005)))
    run.main(str(config(tmp, budget=10.0)))
    assert len(rows(tmp)) == 10
    other = config(tmp)
    other.write_text(other.read_text().replace('base_seed = "s"', 'base_seed = "other"'))
    with pytest.raises(SystemExit):
        run.main(str(other))
