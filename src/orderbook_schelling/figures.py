"""The paper's figures from a run's rows: decoding per pair, sender behaviour, and decoding per word.

usage: uv run python -m orderbook_schelling.figures runs/<name>.jsonl <out_dir>
"""

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from orderbook_schelling.analyze import CONDITIONS, config_order, group, wilson  # noqa: E402

COLOUR = {"unwarned": "#2a78d6", "warned": "#eb6834"}  # validated categorical slots 1-2 (dataviz palette)
STATUS_COLOUR = {
    "sent": "#2a78d6",
    "empty": "#1baf7a",
    "refused": "#eb6834",
    "unparsed": "#eda100",
    "truncated": "#6250d6",
}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update(
    {
        "font.size": 8,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def decoding_by_pair(msgs: list[dict[str, Any]], pairs: list[tuple[str, str]], chance: float, out: Path) -> None:
    """Decoded/all per pair, unwarned and warned side by side, with Wilson 95% bars and the chance level."""
    by = group(msgs, "sender", "receiver", "condition")
    present = [p for p in pairs if any((*p, c) in by for c in CONDITIONS)]
    fig, ax = plt.subplots(figsize=(6.5, 2.4))
    width = 0.38
    for j, cond in enumerate(CONDITIONS):
        for i, (s, r) in enumerate(present):
            rs = by.get((s, r, cond), [])
            if not rs:
                continue
            k, n = sum(x["decoded"] for x in rs), len(rs)
            lo, hi = wilson(k, n)
            x = i + (j - 0.5) * width
            ax.bar(x, k / n, width * 0.92, color=COLOUR[cond], label=cond if i == 0 else None)
            ax.errorbar(x, k / n, yerr=[[max(0.0, k / n - lo)], [max(0.0, hi - k / n)]], color=INK, lw=0.8, capsize=1.5)
    ax.axhline(chance, color=MUTED, lw=0.8, ls="--", label="chance (upper bound)")
    ax.set_xticks(range(len(present)), [f"{s}→{r}" for s, r in present], rotation=40, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("decoded / all trials")
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=3, loc="lower left", bbox_to_anchor=(0, 1.0))
    fig.tight_layout()
    fig.savefig(out / "decoding_by_pair.pdf")
    plt.close(fig)


def sender_status(msgs: list[dict[str, Any]], models: list[str], out: Path) -> None:
    """Share of each sender status per sender model and condition, pooled over receivers."""
    by = group(msgs, "sender", "condition")
    rows = [(s, c) for s in models for c in CONDITIONS if (s, c) in by]
    fig, ax = plt.subplots(figsize=(6.5, 2.0))
    for i, key in enumerate(rows):
        rs, left = by[key], 0.0
        for status, colour in STATUS_COLOUR.items():
            share = sum(r["status"] == status for r in rs) / len(rs)
            if share:
                ax.barh(i, share, left=left, color=colour, edgecolor="white", lw=1, label=status)
            left += share
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles, strict=True))
    ax.legend(unique.values(), unique.keys(), frameon=False, ncol=5, loc="lower left", bbox_to_anchor=(0, 1.0))
    ax.set_yticks(range(len(rows)), [f"{s} ({c})" for s, c in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of sender trials (pooled over receivers)")
    fig.tight_layout()
    fig.savefig(out / "sender_status.pdf")
    plt.close(fig)


def decoding_by_word(msgs: list[dict[str, Any]], pairs: list[tuple[str, str]], words: list[str], out: Path) -> None:
    """Decoded (dark) or not (light) for every word and pair, one panel per condition; blank where no trial exists."""
    by = {(r["sender"], r["receiver"], r["condition"], r["word"]): r["decoded"] for r in msgs}
    present = [p for p in pairs if any((*p, c, w) in by for c in CONDITIONS for w in words)]
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 0.18 * len(present) + 1.2), sharey=True)
    for ax, cond in zip(axes, CONDITIONS, strict=True):
        cells = [[by.get((s, r, cond, w)) for w in words] for s, r in present]
        grid = [[float("nan") if d is None else 1.0 if d else 0.25 for d in row] for row in cells]
        ax.imshow(grid, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.set_title(cond, color=INK)
        ax.set_xticks(range(len(words)), words, rotation=90, fontsize=6)
        ax.set_yticks(range(len(present)), [f"{s}→{r}" for s, r in present], fontsize=6)
        ax.set_xticks([x - 0.5 for x in range(len(words))], minor=True)
        ax.set_yticks([y - 0.5 for y in range(len(present))], minor=True)
        ax.grid(which="minor", color="white", lw=1)
        ax.tick_params(which="minor", length=0)
    fig.tight_layout()
    fig.savefig(out / "decoding_by_word.pdf")
    plt.close(fig)


if __name__ == "__main__":
    rows = [json.loads(line) for line in Path(sys.argv[1]).read_text().splitlines() if line.strip()]
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    models, pairs, words = config_order(rows)
    msgs = [r for r in rows if r["kind"] == "message"]
    clean = [r for r in rows if r["kind"] == "clean"]
    listed = sum(bool(r["guess_in_words"]) for r in clean)
    chance = wilson(listed, len(clean))[1] / len(words) if clean else 0.0
    decoding_by_pair(msgs, pairs, chance, out)
    sender_status(msgs, models, out)
    decoding_by_word(msgs, pairs, words, out)
    print(f"figures written to {out}")
