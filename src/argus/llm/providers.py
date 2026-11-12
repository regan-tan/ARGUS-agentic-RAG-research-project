"""Choose the LLM provider for an experiment run from command-line arguments.

Ollama (local Qwen) stays the default, so existing commands and run metadata are unchanged.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from typing import Any, Literal

from .ollama import DEFAULT_OLLAMA_ENDPOINT, DEFAULT_QWEN_MODEL, OllamaChatClient
from .openai_client import DEFAULT_OPENAI_ENDPOINT, DEFAULT_OPENAI_MODEL, OpenAIChatClient

Provider = Literal["ollama", "openai"]
DEFAULT_MODELS: dict[str, str] = {"ollama": DEFAULT_QWEN_MODEL, "openai": DEFAULT_OPENAI_MODEL}
DEFAULT_ENDPOINTS: dict[str, str] = {
    "ollama": DEFAULT_OLLAMA_ENDPOINT,
    "openai": DEFAULT_OPENAI_ENDPOINT,
}


def add_provider_arguments(parser: argparse.ArgumentParser) -> None:
    """Add ``--provider`` and OpenAI-only options; model and endpoint default per provider."""

    parser.add_argument("--provider", choices=("ollama", "openai"), default="ollama")
    parser.add_argument(
        "--model", default=None, help="Default: the Qwen Ollama tag, or gpt-5.6-luna for OpenAI."
    )
    parser.add_argument("--endpoint", default=None)
    parser.add_argument(
        "--reasoning-effort",
        default="none",
        help="OpenAI only. 'none' keeps hidden reasoning out of the 512-token output budget; "
        "pass 'omit' to send no reasoning setting.",
    )
    parser.add_argument(
        "--openai-schema",
        choices=("strict", "loose"),
        default="strict",
        help="OpenAI only: strict structured outputs (default) or best-effort JSON schema.",
    )
    parser.add_argument(
        "--omit-temperature",
        action="store_true",
        help="OpenAI only: send no temperature, for models that reject it. Recorded in metadata.",
    )
    parser.add_argument("--max-retries", type=int, default=5, help="OpenAI only.")


def resolve_provider_arguments(args: argparse.Namespace) -> None:
    """Fill ``model`` and ``endpoint`` from the provider and reject OpenAI-only misuse."""

    if args.model is None:
        args.model = DEFAULT_MODELS[args.provider]
    if args.endpoint is None:
        args.endpoint = DEFAULT_ENDPOINTS[args.provider]


def build_chat_client(
    args: argparse.Namespace,
    *,
    timeout_seconds: float,
    num_predict: int,
    num_ctx: int,
    top_k: int,
    ollama_client: Callable[..., Any] = OllamaChatClient,
    openai_client: Callable[..., Any] = OpenAIChatClient,
) -> Any:
    """Create the configured client; ``num_predict`` is the output cap for both providers.

    The client classes are parameters so a script can pass its own module-level names, which
    tests replace with fakes.
    """

    if args.provider == "openai":
        return openai_client(
            model=args.model,
            endpoint=args.endpoint,
            timeout_seconds=timeout_seconds,
            temperature=None if args.omit_temperature else 0.0,
            max_output_tokens=num_predict,
            reasoning_effort=None if args.reasoning_effort == "omit" else args.reasoning_effort,
            schema_mode=args.openai_schema,
            max_retries=args.max_retries,
        )
    return ollama_client(
        model=args.model,
        endpoint=args.endpoint,
        timeout_seconds=timeout_seconds,
        num_predict=num_predict,
        num_ctx=num_ctx,
        top_k=top_k,
    )


def generator_identity_metadata(
    provider: str, identity_dict: dict[str, object]
) -> dict[str, object]:
    """Metadata entries naming the generator: unchanged ``ollama`` key for Ollama runs."""

    if provider == "openai":
        return {"provider": "openai", "openai": identity_dict}
    return {"ollama": identity_dict}
