from orderbook_schelling.feed import Order, build_feed, recent_orders, render_tape
from orderbook_schelling.prompts import receiver_prompt, sender_prompt, template


def test_surveillance_sentence_only_in_monitored_prompt() -> None:
    recent = recent_orders("a")
    sentence = template("surveillance")
    assert sentence in sender_prompt("HELLO", recent, monitored=True)
    assert sentence not in sender_prompt("HELLO", recent, monitored=False)


def test_sender_prompt_shows_word_and_recent_orders() -> None:
    recent = recent_orders("a")
    prompt = sender_prompt("HELLO", recent, monitored=False)
    assert "Your message is the word: HELLO." in prompt
    assert render_tape(recent) in prompt
    assert "$" not in prompt


def test_receiver_prompt_ends_with_the_tape_and_never_shows_the_word() -> None:
    tape = render_tape(build_feed([Order("buy", 137.0, 150)], "a"))
    prompt = receiver_prompt(tape)
    assert prompt.endswith(tape)
    assert "HELLO" not in prompt
