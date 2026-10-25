"""Turn raw model replies into orders and words, applying the exchange's order rules."""

import json
import math
import re
from dataclasses import dataclass, field
from typing import Literal

from orderbook_schelling.feed import LAST_PRICE, MAX_QTY, MAX_SENDER_ORDERS, PRICE_COLLAR, Order

SendStatus = Literal["sent", "empty", "unparsed"]  # unparsed: no orders list, e.g. a refusal or malformed reply
RejectReason = Literal[
    "not_an_object", "bad_side", "bad_number", "qty_out_of_range", "price_outside_collar", "over_limit"
]


@dataclass(frozen=True)
class Rejection:
    order: object  # the order exactly as the model wrote it
    reason: RejectReason


@dataclass(frozen=True)
class SenderReply:
    status: SendStatus
    orders: list[Order] = field(default_factory=list)
    rejected: list[Rejection] = field(default_factory=list)


def json_objects(raw: str) -> list[dict[str, object]]:
    """Every JSON object in a reply, in order, ignoring code fences and surrounding text."""
    decoder, found, i = json.JSONDecoder(), [], raw.find("{")
    while i != -1:
        try:
            value, end = decoder.raw_decode(raw, i)
        except json.JSONDecodeError:
            i = raw.find("{", i + 1)
            continue
        if isinstance(value, dict):
            found.append(value)
        i = raw.find("{", end)
    return found


def last_with(raw: str, key: str) -> object:
    """The value of `key` in the last JSON object that has it, or None."""
    return next((obj[key] for obj in reversed(json_objects(raw)) if key in obj), None)


def number(value: object) -> float:
    """A finite number from a JSON number or numeric string; booleans and NaN are not numbers."""
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        raise ValueError("not a number")
    x = float(value)
    if not math.isfinite(x):
        raise ValueError("not finite")
    return x


def check_order(raw: object) -> Order | RejectReason:
    """An order that passes the exchange rules, or the reason it is rejected."""
    if not isinstance(raw, dict):
        return "not_an_object"
    side = str(raw.get("side", "")).strip().lower()
    if side not in ("buy", "sell"):
        return "bad_side"
    try:
        price = round(number(raw["price"]), 2)
        qty = number(raw["qty"])
    except (KeyError, TypeError, ValueError):
        return "bad_number"
    if not qty.is_integer() or not 1 <= qty <= MAX_QTY:
        return "qty_out_of_range"
    if abs(price - LAST_PRICE) > PRICE_COLLAR * LAST_PRICE:
        return "price_outside_collar"
    return Order("buy" if side == "buy" else "sell", price, int(qty))


def parse_sender(raw: str) -> SenderReply:
    """Orders from a sender reply; orders beyond the limit or breaking a rule are rejected with a reason."""
    raw_orders = last_with(raw, "orders")
    if not isinstance(raw_orders, list):
        return SenderReply("unparsed")
    orders: list[Order] = []
    rejected = [Rejection(o, "over_limit") for o in raw_orders[MAX_SENDER_ORDERS:]]
    for o in raw_orders[:MAX_SENDER_ORDERS]:
        result = check_order(o)
        if isinstance(result, Order):
            orders.append(result)
        else:
            rejected.append(Rejection(o, result))
    return SenderReply("sent" if orders else "empty", orders, rejected)


def parse_receiver(raw: str) -> str | None:
    """The receiver's word reduced to upper-case letters (so 'Hello.' scores as HELLO), or None if there is none."""
    word = last_with(raw, "word")
    letters = re.sub(r"[^A-Z]", "", word.upper()) if isinstance(word, str) else ""
    return letters or None
