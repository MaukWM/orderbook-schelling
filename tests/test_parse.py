import json
from pathlib import Path

from orderbook_schelling.feed import Order
from orderbook_schelling.parse import parse_receiver, parse_sender

REPLIES = Path(__file__).parent / "fixtures" / "replies"


def reply(name: str) -> str:
    return (REPLIES / f"{name}.md").read_text()


def test_sender_reply_in_a_code_fence_gives_its_orders() -> None:
    result = parse_sender(reply("sender_orders"))
    assert result.status == "sent" and not result.rejected
    assert [o.qty for o in result.orders] == [150, 37, 412, 9, 260]


def test_refusal_is_unparsed_and_empty_list_is_empty() -> None:
    assert parse_sender(reply("sender_refusal")).status == "unparsed"
    assert parse_sender(reply("sender_empty")).status == "empty"


def test_json_is_found_among_prose_braces_and_fences() -> None:
    raw = (
        'My plan is {x}. Example: {"orders": []}\n'
        '```json\n{"orders": [{"side": "SELL", "price": "137.504", "qty": 150.0}]}\n```\n'
        "I used {my} scheme."
    )
    assert parse_sender(raw).orders == [Order("sell", 137.5, 150)]


def test_orders_breaking_a_rule_are_rejected_with_a_reason() -> None:
    bad = [
        {"side": "hold", "price": 137, "qty": 5},
        {"side": "buy", "price": "x", "qty": 5},
        {"side": "buy", "price": "NaN", "qty": 5},
        {"side": "buy", "price": 137, "qty": True},
        {"side": "buy", "price": 137, "qty": 0},
        {"side": "buy", "price": 137, "qty": 1000},
        {"side": "buy", "price": 137, "qty": 7.5},
        {"side": "buy", "price": 150, "qty": 5},
        "150",
    ]
    result = parse_sender(json.dumps({"orders": bad}))
    assert result.status == "empty" and result.orders == []
    assert [r.reason for r in result.rejected] == [
        "bad_side",
        "bad_number",
        "bad_number",
        "bad_number",
        "qty_out_of_range",
        "qty_out_of_range",
        "qty_out_of_range",
        "price_outside_collar",
        "not_an_object",
    ]


def test_orders_beyond_the_limit_are_rejected() -> None:
    raw = json.dumps({"orders": [{"side": "buy", "price": 137, "qty": k} for k in range(1, 21)]})
    result = parse_sender(raw)
    assert len(result.orders) == 16 and [r.reason for r in result.rejected] == ["over_limit"] * 4


def test_receiver_word_is_reduced_to_letters() -> None:
    assert parse_receiver(reply("receiver_word")) == "RIVER"
    assert parse_receiver('{"word": "Hello."}') == "HELLO"
    assert parse_receiver('{"word": "H E L L O"}') == "HELLO"
    assert parse_receiver("I cannot tell.") is None
    assert parse_receiver('{"word": "  "}') is None
