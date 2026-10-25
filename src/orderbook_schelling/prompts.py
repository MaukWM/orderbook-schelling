"""Sender and receiver prompts, filled from the text templates in templates/."""

from pathlib import Path
from string import Template

from orderbook_schelling.feed import LAST_PRICE, MAX_QTY, MAX_SENDER_ORDERS, PRICE_COLLAR, FeedEntry, render_tape

TEMPLATES = Path(__file__).parent / "templates"


def template(name: str) -> str:
    """The text of templates/<name>.md without its trailing newline."""
    return (TEMPLATES / f"{name}.md").read_text().rstrip("\n")


def sender_prompt(word: str, recent: list[FeedEntry], warned: bool) -> str:
    """The sender's prompt: setting, the recent feed, the word, and the order format."""
    return Template(template("sender")).substitute(
        last_price=LAST_PRICE,
        surveillance=" " + template("surveillance") if warned else "",
        word=word,
        n_recent=len(recent),
        recent=render_tape(recent),
        max_orders=MAX_SENDER_ORDERS,
        max_qty=MAX_QTY,
        collar_pct=round(PRICE_COLLAR * 100),
    )


def receiver_prompt(tape: str) -> str:
    """The receiver's prompt: the task and the anonymous tape."""
    return Template(template("receiver")).substitute(tape=tape)
