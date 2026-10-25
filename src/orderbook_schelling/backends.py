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
TIMEOUT_S = 1800.0  # long reasoning at 16k output tokens can exceed the SDK default of 600 s


@dataclass(frozen=True)
class Reply:
    text: str
    stop_reason: str | None  # e.g. end_turn / stop / max_tokens / length / refusal / content_filter
    model_served: str  # the model id the provider reports, not the requested alias
    provider: str  # anthropic, openai, or the upstream provider OpenRouter used
    input_tokens: int
    output_tokens: int  # includes reasoning tokens
    reasoning_tokens: int | None = None  # the reasoning part of output_tokens, where the API reports it
    reasoning: str | None = None  # reasoning text the API returns (Anthropic thinking, OpenRouter reasoning)
    native_stop_reason: str | None = None  # the upstream provider's own stop reason (OpenRouter)
    reported_cost_usd: float | None = None  # cost billed as reported by OpenRouter
    error: str | None = None  # API failure after the SDK's retries; never a refusal


@cache
def anthropic_client() -> anthropic.Anthropic:
    load_dotenv()
    return anthropic.Anthropic(max_retries=SDK_RETRIES, timeout=TIMEOUT_S)


@cache
def openai_client(route: Route) -> openai.OpenAI:
    load_dotenv()
    if route == "openrouter":
        key = os.environ["OPENROUTER_API_KEY"]
        return openai.OpenAI(base_url=OPENROUTER_URL, api_key=key, max_retries=SDK_RETRIES, timeout=TIMEOUT_S)
    return openai.OpenAI(max_retries=SDK_RETRIES, timeout=TIMEOUT_S)


def ask_anthropic(model: str, prompt: str, params: dict[str, Any]) -> Reply:
    """Messages API, streamed so long reasoning cannot time out; thinking blocks are kept apart from the reply text."""
    with anthropic_client().messages.stream(
        model=model, messages=[{"role": "user", "content": prompt}], **params
    ) as stream:
        msg = stream.get_final_message()
    text = "".join(block.text for block in msg.content if block.type == "text")
    thinking = "\n".join(block.thinking for block in msg.content if block.type == "thinking") or None
    details = msg.usage.output_tokens_details
    return Reply(
        text,
        msg.stop_reason,
        msg.model,
        "anthropic",
        msg.usage.input_tokens,
        msg.usage.output_tokens,
        reasoning_tokens=getattr(details, "thinking_tokens", None),
        reasoning=thinking,
    )


def ask_openai_compatible(route: Route, model: str, prompt: str, params: dict[str, Any]) -> Reply:
    """Chat Completions on OpenAI or OpenRouter; an OpenRouter error inside a 200 response is raised as an error."""
    messages: list[Any] = [{"role": "user", "content": prompt}]
    if route == "openrouter":  # OpenRouter-specific fields (e.g. reasoning) are not SDK arguments
        r = openai_client(route).chat.completions.create(model=model, messages=messages, extra_body=params)
    else:
        r = openai_client(route).chat.completions.create(model=model, messages=messages, **params)
    if getattr(r, "error", None) or not r.choices or r.choices[0].finish_reason == "error":
        raise RuntimeError(f"provider error: {getattr(r, 'error', None) or 'finish_reason error'}")
    choice = r.choices[0]
    usage = r.usage
    details = usage.completion_tokens_details if usage else None
    return Reply(
        choice.message.content or "",
        "refusal" if choice.message.refusal else choice.finish_reason,
        r.model,
        getattr(r, "provider", None) or route,
        usage.prompt_tokens if usage else 0,
        usage.completion_tokens if usage else 0,
        reasoning_tokens=details.reasoning_tokens if details else None,
        reasoning=getattr(choice.message, "reasoning", None),
        native_stop_reason=getattr(choice, "native_finish_reason", None),
        reported_cost_usd=getattr(usage, "cost", None),
    )


def ask(route: Route, model: str, prompt: str, params: dict[str, Any]) -> Reply:
    """Call a model with the given request parameters; any failure comes back as an error reply."""
    try:
        if route == "anthropic":
            return ask_anthropic(model, prompt, params)
        return ask_openai_compatible(route, model, prompt, params)
    except Exception as e:  # API failures and malformed responses both become a retryable error reply
        return Reply("", None, model, route, 0, 0, error=f"{type(e).__name__}: {str(e)[:300]}")
