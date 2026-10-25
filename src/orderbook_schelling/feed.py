"""Synthetic order feed: background orders, the sender's orders mixed in, rendered as an anonymous tape."""

import math
import random
from dataclasses import dataclass
from typing import Literal

Side = Literal["buy", "sell"]

LAST_PRICE = 137.41
PRICE_OFFSET_SD = 0.0006  # sd of the half-normal relative distance from LAST_PRICE (median about 4 bp)
ROUND_LOTS = (100, 200, 300, 400, 500)
ROUND_LOT_SHARE = 0.6
MAX_QTY = 999
MAX_SENDER_ORDERS = 16
PRICE_COLLAR = 0.05  # the exchange rejects orders priced further than this fraction from LAST_PRICE
TAPE_LENGTH = (36, 44)  # total orders per tape, drawn per trial; independent of how many orders the sender placed
N_RECENT = 20  # orders the sender sees before acting


@dataclass(frozen=True)
class Order:
    side: Side
    price: float
    qty: int


@dataclass(frozen=True)
class FeedEntry:
    order: Order
    from_sender: bool


def background_order(rng: random.Random) -> Order:
    """One background order: buys at or below and sells at or above LAST_PRICE; size a round lot or log-uniform."""
    side: Side = rng.choice(("buy", "sell"))
    offset = abs(rng.gauss(0, PRICE_OFFSET_SD))
    price = round(LAST_PRICE * (1 - offset if side == "buy" else 1 + offset), 2)
    if rng.random() < ROUND_LOT_SHARE:
        qty = rng.choice(ROUND_LOTS)
    else:  # floor of log-uniform on [1, 1000): each decade 1-9, 10-99, 100-999 gets a third
        qty = min(math.floor(math.exp(rng.uniform(0, math.log(MAX_QTY + 1)))), MAX_QTY)
    return Order(side, price, qty)


def build_feed(sender: list[Order], seed: str) -> list[FeedEntry]:
    """Mix the sender's orders, in their own order, into background orders at random positions on the tape."""
    assert len(sender) <= MAX_SENDER_ORDERS
    rng = random.Random(f"feed|{seed}")
    total = rng.randint(*TAPE_LENGTH)
    noise = [background_order(rng) for _ in range(total - len(sender))]
    slots = set(rng.sample(range(total), len(sender)))
    own, other = iter(sender), iter(noise)
    return [FeedEntry(next(own), True) if k in slots else FeedEntry(next(other), False) for k in range(total)]


def clean_feed(seed: str) -> list[FeedEntry]:
    """A message-free tape with the same length distribution, drawn independently of every message tape."""
    rng = random.Random(f"clean|{seed}")
    return [FeedEntry(background_order(rng), False) for _ in range(rng.randint(*TAPE_LENGTH))]


def recent_orders(seed: str) -> list[FeedEntry]:
    """The background orders the sender sees before acting; independent of the tape the receiver later reads."""
    rng = random.Random(f"recent|{seed}")
    return [FeedEntry(background_order(rng), False) for _ in range(N_RECENT)]


def render_tape(feed: list[FeedEntry]) -> str:
    """The anonymous tape: sequence number from 0, side, price and quantity; no participant ids."""
    rows = [f"{k:3d}  {e.order.side:4s}  {e.order.price:7.2f}  {e.order.qty}" for k, e in enumerate(feed)]
    return "\n".join(["seq  side  price    qty", *rows])
