"""Minimal typed integration with the local Ollama HTTP API."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_OLLAMA_ENDPOINT = "http://localhost:11434"
DEFAULT_QWEN_MODEL = "qwen3:4b-instruct-2507-q4_K_M"


class OllamaError(RuntimeError):
    """Base error for an Ollama interaction."""


class OllamaTransportError(OllamaError):
    """Raised when the local Ollama service cannot be contacted."""


class OllamaMissingModelError(OllamaError):
    """Raised when the configured local model is not installed."""


class OllamaResponseError(OllamaError):
    """Raised when Ollama returns an unexpected or invalid API response."""


@dataclass(frozen=True)
class GenerationMetadata:
    """Measured usage and timing information returned by Ollama where available."""

    prompt_tokens: int | None
    completion_tokens: int | None
    total_duration_ns: int | None
    load_duration_ns: int | None
    prompt_eval_duration_ns: int | None
    eval_duration_ns: int | None
    wall_clock_seconds: float

    def to_dict(self) -> dict[str, int | float | None]:
        return asdict(self)


@dataclass(frozen=True)
class OllamaChatResponse:
    """One non-streaming chat completion and its observed metadata."""

    content: str
    metadata: GenerationMetadata


@dataclass(frozen=True)
class OllamaModelIdentity:
    """Available reproducibility identity returned by the local Ollama runtime."""

    model: str
    digest: str | None
    runtime_version: str | None
    details: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "model": self.model,
            "digest": self.digest,
            "runtime_version": self.runtime_version,
            "details": self.details,
        }


JsonRequest = Callable[[str, str, Mapping[str, object] | None, float], Mapping[str, Any]]


def _optional_int(response: Mapping[str, Any], field: str) -> int | None:
    value = response.get(field)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _http_json_request(
    endpoint: str, method: str, path: str, payload: Mapping[str, object] | None, timeout: float
) -> Mapping[str, Any]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(
        f"{endpoint.rstrip('/')}{path}",
        data=body,
        method=method,
        headers={"Content-Type": "application/json"} if body is not None else {},
    )
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310 - configured local endpoint
            raw = response.read().decode("utf-8")
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        message = f"Ollama returned HTTP {error.code}: {detail}"
        if error.code == 404 and "model" in detail.lower():
            raise OllamaMissingModelError(message) from error
        raise OllamaResponseError(message) from error
    except (URLError, TimeoutError, OSError) as error:
        raise OllamaTransportError(f"Could not reach Ollama at {endpoint}: {error}") from error
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError as error:
        raise OllamaResponseError("Ollama returned malformed JSON") from error
    if not isinstance(decoded, dict):
        raise OllamaResponseError("Ollama returned a JSON value other than an object")
    return decoded


class OllamaChatClient:
    """One configured local Ollama model, with deliberately narrow chat support."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_QWEN_MODEL,
        endpoint: str = DEFAULT_OLLAMA_ENDPOINT,
        timeout_seconds: float = 300.0,
        temperature: float = 0.0,
        num_predict: int = 512,
        num_ctx: int = 8_192,
        top_k: int = 40,
        request: JsonRequest | None = None,
    ) -> None:
        if timeout_seconds <= 0 or num_predict <= 0 or num_ctx <= 0 or top_k <= 0:
            raise ValueError("Ollama timeout and generation limits must be positive")
        self.model = model
        self.endpoint = endpoint.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.options: dict[str, int | float] = {
            "temperature": temperature,
            "num_predict": num_predict,
            "num_ctx": num_ctx,
            "top_k": top_k,
        }
        self._request = request or (
            lambda method, path, payload, timeout: _http_json_request(
                self.endpoint, method, path, payload, timeout
            )
        )

    def model_identity(self) -> OllamaModelIdentity:
        """Verify the exact installed tag and return the locally available identity."""

        tags = self._request("GET", "/api/tags", None, self.timeout_seconds)
        models = tags.get("models")
        if not isinstance(models, list):
            raise OllamaResponseError("Ollama /api/tags response has no models list")
        matching = next(
            (
                entry
                for entry in models
                if isinstance(entry, dict) and entry.get("name") == self.model
            ),
            None,
        )
        if matching is None:
            raise OllamaMissingModelError(
                f"Configured Ollama model is not installed: {self.model}. "
                "Install it in Ollama; ARGUS will not pull a substitute."
            )
        version_response = self._request("GET", "/api/version", None, self.timeout_seconds)
        details = matching.get("details")
        return OllamaModelIdentity(
            model=self.model,
            digest=matching.get("digest") if isinstance(matching.get("digest"), str) else None,
            runtime_version=(
                version_response.get("version")
                if isinstance(version_response.get("version"), str)
                else None
            ),
            details=dict(details) if isinstance(details, dict) else {},
        )

    def chat(
        self, *, system: str, user: str, response_schema: Mapping[str, object]
    ) -> OllamaChatResponse:
        """Make exactly one non-streaming structured chat request."""

        payload: dict[str, object] = {
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "format": dict(response_schema),
            "options": self.options,
        }
        started = time.perf_counter()
        response = self._request("POST", "/api/chat", payload, self.timeout_seconds)
        elapsed = time.perf_counter() - started
        message = response.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise OllamaResponseError("Ollama chat response is missing message.content")
        return OllamaChatResponse(
            content=message["content"],
            metadata=GenerationMetadata(
                prompt_tokens=_optional_int(response, "prompt_eval_count"),
                completion_tokens=_optional_int(response, "eval_count"),
                total_duration_ns=_optional_int(response, "total_duration"),
                load_duration_ns=_optional_int(response, "load_duration"),
                prompt_eval_duration_ns=_optional_int(response, "prompt_eval_duration"),
                eval_duration_ns=_optional_int(response, "eval_duration"),
                wall_clock_seconds=elapsed,
            ),
        )
