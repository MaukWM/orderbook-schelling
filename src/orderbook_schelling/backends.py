"""One model call through the Anthropic API, the OpenAI API, or OpenRouter; keys come from the environment (.env)."""

import os
from dataclasses import dataclass
from functools import cache
from typing import Any, Literal

import anthropic
import openai
from dotenv import load_dotenv

Route = Literal["anthropic", "openai", "openrouter"]
OPENROUTER_URL = "https://openrouter.ai/api/v1"
SDK_RETRIES = 4  # the SDKs retry rate limits, timeouts and 5xx with backoff


@dataclass(frozen=True)
class Reply:
    text: str
    stop_reason: str | None  # e.g. end_turn / stop / max_tokens / length / refusal
    model_served: str  # the model id the provider reports, not the requested alias
    provider: str  # anthropic, openai, or the upstream provider OpenRouter used
    input_tokens: int
    output_tokens: int
    error: str | None = None  # API failure after the SDK's retries; never a refusal


@cache
def anthropic_client() -> anthropic.Anthropic:
    load_dotenv()
    return anthropic.Anthropic(max_retries=SDK_RETRIES)


@cache
def openai_client(route: Route) -> openai.OpenAI:
    load_dotenv()
    if route == "openrouter":
        return openai.OpenAI(base_url=OPENROUTER_URL, api_key=os.environ["OPENROUTER_API_KEY"], max_retries=SDK_RETRIES)
    return openai.OpenAI(max_retries=SDK_RETRIES)


def ask_anthropic(model: str, prompt: str, params: dict[str, Any]) -> Reply:
    """Messages API, streamed so long reasoning cannot time out; only the text blocks form the reply."""
    with anthropic_client().messages.stream(
        model=model, messages=[{"role": "user", "content": prompt}], **params
    ) as stream:
        msg = stream.get_final_message()
    text = "".join(block.text for block in msg.content if block.type == "text")
    usage = msg.usage
    return Reply(text, msg.stop_reason, msg.model, "anthropic", usage.input_tokens, usage.output_tokens)


def ask_openai_compatible(route: Route, model: str, prompt: str, params: dict[str, Any]) -> Reply:
    """Chat Completions on OpenAI or OpenRouter; OpenRouter also reports which upstream provider answered."""
    messages: list[Any] = [{"role": "user", "content": prompt}]
    if route == "openrouter":  # OpenRouter-specific fields (e.g. reasoning) are not SDK arguments
        r = openai_client(route).chat.completions.create(model=model, messages=messages, extra_body=params)
    else:
        r = openai_client(route).chat.completions.create(model=model, messages=messages, **params)
    choice = r.choices[0]
    usage = r.usage
    stop = "refusal" if choice.message.refusal else choice.finish_reason
    provider = getattr(r, "provider", None) or route
    return Reply(
        choice.message.content or "",
        stop,
        r.model,
        provider,
        usage.prompt_tokens if usage else 0,
        usage.completion_tokens if usage else 0,
    )


def ask(route: Route, model: str, prompt: str, params: dict[str, Any]) -> Reply:
    """Call a model with the given request parameters; any failure comes back as an error reply."""
    try:
        if route == "anthropic":
            return ask_anthropic(model, prompt, params)
        return ask_openai_compatible(route, model, prompt, params)
    except Exception as e:  # API failures and malformed responses both become a retryable error reply
        return Reply("", None, model, route, 0, 0, error=f"{type(e).__name__}: {str(e)[:300]}")
