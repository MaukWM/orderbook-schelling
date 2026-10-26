"""The paper's figures from a run's rows: decoding per pair, sender behaviour, and decoding per word.

usage: uv run python -m orderbook_schelling.figures runs/<name>.jsonl <out_dir>
"""

import json
import sys
import textwrap
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
    ys = [i + 0.6 * models.index(s) for i, (s, _) in enumerate(rows)]  # a gap between sender models
    fig, ax = plt.subplots(figsize=(6.5, 2.4))
    for y, key in zip(ys, rows, strict=True):
        rs, left = by[key], 0.0
        for status, colour in STATUS_COLOUR.items():
            share = sum(r["status"] == status for r in rs) / len(rs)
            if share:
                ax.barh(y, share, left=left, color=colour, edgecolor="white", lw=1, label=status)
            left += share
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles, strict=True))
    ax.legend(unique.values(), unique.keys(), frameon=False, ncol=5, loc="lower left", bbox_to_anchor=(0, 1.0))
    ax.set_yticks(ys, [f"{s} ({c})" for s, c in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of sender trials (pooled over receivers)")
    fig.tight_layout()
    # left-align the labels: pad them by the widest label's width
    fig.canvas.draw()
    widest = max(t.get_window_extent().width for t in ax.get_yticklabels()) * 72 / fig.dpi
    ax.tick_params(axis="y", pad=widest + 3)
    for t in ax.get_yticklabels():
        t.set_horizontalalignment("left")
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


EXAMPLE = ("astra", "astra", "HORSE", "unwarned")  # sender, receiver, word, condition of the example trial
HIGHLIGHT, ROBOT = "#fbd77a", "#a9bfdc"


def robot(ax: Any, x: float, y: float) -> None:
    """A small robot head centred at (x, y) in axes data units."""
    from matplotlib.patches import Ellipse, FancyBboxPatch

    ax.add_patch(FancyBboxPatch((x - 0.42, y - 0.25), 0.84, 0.5, boxstyle="round,pad=0.02", fc=ROBOT, ec=INK, lw=0.8))
    for dx in (-0.18, 0.18):
        ax.add_patch(Ellipse((x + dx, y + 0.06), 0.17, 0.14, fc="white", ec=INK, lw=0.6))
    ax.plot([x - 0.16, x + 0.16], [y - 0.13, y - 0.13], color=INK, lw=0.8)
    ax.plot([x, x], [y + 0.27, y + 0.4], color=INK, lw=0.8)
    ax.add_patch(Ellipse((x, y + 0.42), 0.09, 0.075, fc=INK))


def example_trial(msgs: list[dict[str, Any]], out: Path) -> None:
    """The example trial: the sender's word, the tape with its orders highlighted, the receiver's answer."""
    sender, receiver, word, cond = EXAMPLE
    r = next(r for r in msgs if (r["sender"], r["receiver"], r["word"], r["condition"]) == EXAMPLE)
    tape = [(line.split(), mine) for line, mine in zip(r["tape"].splitlines()[1:], r["from_sender"], strict=True)]
    half = (len(tape) + 1) // 2
    top = 0.15 * half + 0.5
    fig, ax = plt.subplots(figsize=(6.5, 0.12 * half + 0.35))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, top)
    ax.axis("off")
    mid = top / 2
    robot(ax, 0.8, mid)
    ax.text(0.8, mid + 0.6, f"sender ({sender})\nword: {word}", fontsize=7, ha="center", va="bottom", color=INK)
    for col in range(2):
        ax.text(2.0 + 2.6 * col, top - 0.15, "seq side price    qty", fontsize=6.2, family="monospace", color=MUTED)
    for i, (cells, mine) in enumerate(tape):
        qty = int(cells[3])
        letter = f" {chr(qty)}" if mine and 65 <= qty <= 90 else "  "
        ax.text(
            2.0 + 2.6 * (i // half),
            top - 0.32 - 0.15 * (i % half),
            f"{cells[0]:>3} {cells[1]:<4} {cells[2]} {cells[3]:>4}{letter}",
            fontsize=6.2,
            family="monospace",
            va="center",
            color=INK,
            weight="bold" if letter.strip() else "normal",
            bbox={"fc": HIGHLIGHT, "ec": "none", "pad": 0.4} if mine else None,
        )
    arrow = {"arrowstyle": "->", "color": MUTED}
    ax.annotate("", (1.95, mid), (1.45, mid), arrowprops=arrow)
    ax.annotate("", (7.6, mid), (7.1, mid), arrowprops=arrow)
    robot(ax, 8.4, mid)
    ax.text(8.4, mid + 0.6, f"receiver ({receiver})", fontsize=7, ha="center", va="bottom", color=INK)
    ax.text(8.4, mid - 0.5, f'answer: "{r["guess"]}"', fontsize=7.5, ha="center", va="top", weight="bold", color=INK)
    how = json.loads(r["receiver_reply"]["text"])["how"]
    ax.text(8.4, mid - 0.72, textwrap.fill(how, 30), fontsize=6, ha="center", va="top", style="italic", color=MUTED)
    fig.tight_layout(pad=0.1)
    fig.savefig(out / "example_trial.pdf")
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
    example_trial(msgs, out)
    print(f"figures written to {out}")
