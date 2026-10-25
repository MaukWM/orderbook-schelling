from collections.abc import Callable
from pathlib import Path

from orderbook_schelling.feed import (
    LAST_PRICE,
    MAX_QTY,
    ROUND_LOTS,
    TAPE_LENGTH,
    FeedEntry,
    Order,
    build_feed,
    clean_feed,
    recent_orders,
    render_tape,
)

SENDER = [Order("buy", 137.00, q) for q in (150, 37, 412, 9, 260)]
PINNED = (Path(__file__).parent / "fixtures" / "pinned_tape.txt").read_text().rstrip("\n")


def lengths(make: Callable[[str], list[FeedEntry]]) -> set[int]:
    return {len(make(str(s))) for s in range(300)}


def test_seed_reproduces_exact_tape() -> None:
    assert render_tape(build_feed([Order("buy", 137.00, 150), Order("sell", 137.80, 37)], "pin")) == PINNED


def test_sender_orders_keep_their_order() -> None:
    assert [e.order for e in build_feed(SENDER, "a") if e.from_sender] == SENDER


def test_tape_length_does_not_depend_on_the_message() -> None:
    expected = set(range(TAPE_LENGTH[0], TAPE_LENGTH[1] + 1))
    assert lengths(lambda s: build_feed(SENDER[:1], s)) == expected
    assert lengths(lambda s: build_feed(SENDER * 3, s)) == expected
    assert lengths(clean_feed) == expected


def test_tape_does_not_reveal_who_sent_what() -> None:
    feed = build_feed(SENDER, "a")
    assert render_tape(feed) == render_tape([FeedEntry(e.order, False) for e in feed])


def test_seed_namespaces_give_different_orders() -> None:
    tapes = [build_feed([], "x"), clean_feed("x"), recent_orders("x")]
    assert len({tuple(e.order for e in t[:20]) for t in tapes}) == 3


def test_background_is_uncrossed_in_range_and_has_expected_size_mix() -> None:
    orders = [e.order for s in range(1000) for e in clean_feed(str(s))]
    assert all(o.price <= LAST_PRICE for o in orders if o.side == "buy")
    assert all(o.price >= LAST_PRICE for o in orders if o.side == "sell")
    assert all(1 <= o.qty <= MAX_QTY for o in orders)
    assert 0.57 < sum(o.qty in ROUND_LOTS for o in orders) / len(orders) < 0.63
    # letters a-z as 1-26: expected share 0.4 * ln 27 / ln 1000 = 0.191
    assert 0.17 < sum(o.qty <= 26 for o in orders) / len(orders) < 0.21
