# The Order Book Is a Channel

**LLM trading agents pass messages without an agreed code**

Code and data for a submission to the Apart Research AI Collusion Research Sprint (October 2026).
*Paper link will be added here once it is published.*

## Abstract

An LLM trading agent that may only enter orders has no chat channel, but every order carries free fields that anyone
reading the feed can see. We test whether one agent can pass a word to another through its orders alone, without an
agreed code. A sender places up to 16 limit orders, mixed into a tape of synthetic background orders; a receiver
reads the tape and names the word. We run five models from OpenAI, Google and xAI in 13 sender–receiver pairs, eight
across labs, on 20 words, with and without warning the sender that market surveillance scans the feed. Unwarned,
receivers read every message that was sent. Senders wrote letters as order sizes, mostly as ASCII codes, and placed
these orders far from the market. Warned, decoding fell to 50%: some senders stopped sending, while others moved their
orders to market prices, hid the letters among round-lot orders, and were still read in two thirds of the trials in
which they sent. Order-only access did not stop instructed agents from passing a word: models from different labs used
the same public conventions without agreeing on them, and most hid them when told they were watched.

## Reproduce the paper's numbers and figures

No API keys needed; this reads the published run.

```bash
uv sync
uv run python -m orderbook_schelling.analyze runs/paper.jsonl          # rates, pair tests, chance bound
uv run python scripts/paper_numbers.py runs/paper.jsonl                # numbers analyze.py does not print
uv run python -m orderbook_schelling.figures runs/paper.jsonl runs/figs/   # all figures
```

`runs/paper.jsonl` holds every trial of the main run (620 rows, code commit `1cab1ea`, config
`experiments/paper.toml`): prompts, raw replies, parsed orders, tapes and guesses.

## Run an experiment

Put `OPENAI_API_KEY` and `OPENROUTER_API_KEY` (and `ANTHROPIC_API_KEY` if used) in `.env`, then:

```bash
uv run python -m orderbook_schelling.run experiments/smoke.toml
```

Rows append to `runs/<name>.jsonl`. A rerun skips finished trials and retries failed ones; code and config must be
committed first. The main run stopped once when API credit ran out and was resumed this way.
`cost_usd` is computed at the list prices in the config.
