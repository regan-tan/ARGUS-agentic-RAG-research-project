"""Minimal typed client for the OpenAI Responses API, interchangeable with the Ollama client.

The client exposes the same ``chat(system=..., user=..., response_schema=...)`` method as
``OllamaChatClient`` so pipelines and agents need no changes. It uses only the standard library.
The API key is read from the ``OPENAI_API_KEY`` environment variable and is never stored in
metadata or outputs.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .ollama import GenerationMetadata, OllamaChatResponse

DEFAULT_OPENAI_ENDPOINT = "https://api.openai.com/v1"
DEFAULT_OPENAI_MODEL = "gpt-5.6-luna"
OPENAI_API_KEY_VARIABLE = "OPENAI_API_KEY"
RETRYABLE_STATUS_CODES = frozenset({408, 409, 429, 500, 502, 503, 504})
SchemaMode = Literal["strict", "loose"]


class OpenAIError(RuntimeError):
    """Base error for an OpenAI interaction."""


class OpenAIConfigurationError(OpenAIError):
    """Raised when the client is not configured to make a request (for example, no API key)."""


class OpenAITransportError(OpenAIError):
    """Raised when the API cannot be reached; requests failing this way may be retried."""


class OpenAIResponseError(OpenAIError):
    """Raised when the API returns an error status or an unexpected response body."""

    def __init__(
        self, message: str, *, status_code: int | None = None, retry_after: float | None = None
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after


@dataclass(frozen=True)
class OpenAIGenerationMetadata(GenerationMetadata):
    """Ollama-compatible usage metadata plus the fields only the OpenAI API reports."""

    response_model: str | None = None
    response_status: str | None = None
    reasoning_tokens: int | None = None
    retries: int = 0


@dataclass(frozen=True)
class OpenAIModelIdentity:
    """Reproducibility identity for an API model. API models have no local digest."""

    model: str
    resolved_model: str | None
    owned_by: str | None
    created: int | None
    settings: dict[str, object]
    digest: None = None
    runtime_version: str = "openai-responses-api"

    def to_dict(self) -> dict[str, object]:
        return {
            "provider": "openai",
            "model": self.model,
            "resolved_model": self.resolved_model,
            "owned_by": self.owned_by,
            "created": self.created,
            "runtime_version": self.runtime_version,
            "settings": self.settings,
        }


JsonRequest = Callable[[str, str, Mapping[str, object] | None, float], Mapping[str, Any]]


def strict_json_schema(schema: Mapping[str, object]) -> dict[str, object]:
    """Return a copy of ``schema`` that satisfies OpenAI strict structured outputs.

    Strict mode requires ``additionalProperties: false`` and every property listed in
    ``required`` on each object. ARGUS schemas already require every property, so the
    conversion only adds ``additionalProperties``; an optional property is rejected rather than
    silently made required.
    """

    def convert(node: object) -> object:
        if isinstance(node, Mapping):
            converted = {key: convert(value) for key, value in node.items()}
            properties = converted.get("properties")
            if converted.get("type") == "object" and isinstance(properties, Mapping):
                required = converted.get("required", [])
                missing = sorted(set(properties).difference(required))  # type: ignore[arg-type]
                if missing:
                    raise ValueError(
                        f"strict schema mode cannot express optional properties: {missing}"
                    )
                converted["additionalProperties"] = False
            return converted
        if isinstance(node, list):
            return [convert(value) for value in node]
        return node

    result = convert(schema)
    assert isinstance(result, dict)
    return result


def _optional_int(mapping: Mapping[str, Any] | None, field: str) -> int | None:
    if not isinstance(mapping, Mapping):
        return None
    value = mapping.get(field)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _retry_after_seconds(error: HTTPError) -> float | None:
    value = error.headers.get("Retry-After") if error.headers else None
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None


def _http_json_request(
    endpoint: str,
    api_key: str,
    method: str,
    path: str,
    payload: Mapping[str, object] | None,
    timeout: float,
) -> Mapping[str, Any]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Authorization": f"Bearer {api_key}"}
    if body is not None:
        headers["Content-Type"] = "application/json"
    request = Request(f"{endpoint.rstrip('/')}{path}", data=body, method=method, headers=headers)
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310 - configured API endpoint
            raw = response.read().decode("utf-8")
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise OpenAIResponseError(
            f"OpenAI returned HTTP {error.code}: {detail}",
            status_code=error.code,
            retry_after=_retry_after_seconds(error),
        ) from error
    except (URLError, TimeoutError, OSError) as error:
        raise OpenAITransportError(f"Could not reach OpenAI at {endpoint}: {error}") from error
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError as error:
        raise OpenAIResponseError("OpenAI returned malformed JSON") from error
    if not isinstance(decoded, dict):
        raise OpenAIResponseError("OpenAI returned a JSON value other than an object")
    return decoded


def _output_text(response: Mapping[str, Any]) -> str:
    """Concatenate the assistant's output text; raise on a refusal or a missing message."""

    output = response.get("output")
    if not isinstance(output, list):
        raise OpenAIResponseError("OpenAI response has no output list")
    texts: list[str] = []
    for item in output:
        if not isinstance(item, Mapping) or item.get("type") != "message":
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, Mapping):
                continue
            if part.get("type") == "refusal":
                raise OpenAIResponseError(f"OpenAI refused the request: {part.get('refusal')}")
            if part.get("type") == "output_text" and isinstance(part.get("text"), str):
                texts.append(part["text"])
    return "".join(texts)


class OpenAIChatClient:
    """One configured OpenAI model with the same narrow interface as ``OllamaChatClient``.

    Generation settings mirror the Ollama runs where the API allows: temperature 0 and a
    512-token output cap. ``reasoning_effort="none"`` keeps hidden reasoning tokens from using
    the output budget. Transient HTTP errors (rate limits, server errors) and connection errors
    are retried with backoff; any other error is raised to the caller, which records it as a
    failed attempt exactly as for Ollama.
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_OPENAI_MODEL,
        endpoint: str = DEFAULT_OPENAI_ENDPOINT,
        timeout_seconds: float = 300.0,
        temperature: float | None = 0.0,
        max_output_tokens: int = 512,
        reasoning_effort: str | None = "none",
        schema_mode: SchemaMode = "strict",
        max_retries: int = 5,
        retry_base_seconds: float = 2.0,
        api_key: str | None = None,
        request: JsonRequest | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if timeout_seconds <= 0 or max_output_tokens <= 0:
            raise ValueError("OpenAI timeout and output-token limit must be positive")
        if max_retries < 0 or retry_base_seconds < 0:
            raise ValueError("OpenAI retry settings must be non-negative")
        if schema_mode not in {"strict", "loose"}:
            raise ValueError("schema_mode must be 'strict' or 'loose'")
        self.model = model
        self.endpoint = endpoint.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.reasoning_effort = reasoning_effort
        self.schema_mode: SchemaMode = schema_mode
        self.max_retries = max_retries
        self.retry_base_seconds = retry_base_seconds
        self._api_key = api_key
        self._sleep = sleep
        self._request = request or self._default_request

    def settings(self) -> dict[str, object]:
        """Generation settings recorded in run metadata (never the API key)."""

        return {
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
            "reasoning_effort": self.reasoning_effort,
            "schema_mode": self.schema_mode,
            "max_retries": self.max_retries,
            "retryable_status_codes": sorted(RETRYABLE_STATUS_CODES),
            "store": False,
        }

    def _key(self) -> str:
        key = self._api_key or os.environ.get(OPENAI_API_KEY_VARIABLE)
        if not key:
            raise OpenAIConfigurationError(
                f"Set the {OPENAI_API_KEY_VARIABLE} environment variable to use OpenAI models"
            )
        return key

    def _default_request(
        self, method: str, path: str, payload: Mapping[str, object] | None, timeout: float
    ) -> Mapping[str, Any]:
        return _http_json_request(self.endpoint, self._key(), method, path, payload, timeout)

    def _with_retries(
        self, method: str, path: str, payload: Mapping[str, object] | None
    ) -> tuple[Mapping[str, Any], int]:
        attempt = 0
        while True:
            try:
                return self._request(method, path, payload, self.timeout_seconds), attempt
            except (OpenAITransportError, OpenAIResponseError) as error:
                status = getattr(error, "status_code", None)
                retryable = isinstance(error, OpenAITransportError) or (
                    status in RETRYABLE_STATUS_CODES
                )
                if not retryable or attempt >= self.max_retries:
                    raise
                delay = getattr(error, "retry_after", None)
                if delay is None:
                    delay = min(self.retry_base_seconds * (2**attempt), 60.0)
                self._sleep(delay)
                attempt += 1

    def model_identity(self) -> OpenAIModelIdentity:
        """Check the key and model access, and return the identity recorded in metadata."""

        self._key()
        info, _ = self._with_retries("GET", f"/models/{self.model}", None)
        resolved = info.get("id")
        owned_by = info.get("owned_by")
        return OpenAIModelIdentity(
            model=self.model,
            resolved_model=resolved if isinstance(resolved, str) else None,
            owned_by=owned_by if isinstance(owned_by, str) else None,
            created=_optional_int(info, "created"),
            settings=self.settings(),
        )

    def request_payload(
        self, *, system: str, user: str, response_schema: Mapping[str, object]
    ) -> dict[str, object]:
        """Build the Responses API body; exposed so tests can check it without the network."""

        schema = (
            strict_json_schema(response_schema)
            if self.schema_mode == "strict"
            else dict(response_schema)
        )
        payload: dict[str, object] = {
            "model": self.model,
            "instructions": system,
            "input": [{"role": "user", "content": user}],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "argus_output",
                    "schema": schema,
                    "strict": self.schema_mode == "strict",
                }
            },
            "max_output_tokens": self.max_output_tokens,
            "store": False,
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.reasoning_effort is not None:
            payload["reasoning"] = {"effort": self.reasoning_effort}
        return payload

    def chat(
        self, *, system: str, user: str, response_schema: Mapping[str, object]
    ) -> OllamaChatResponse:
        """Make one structured request (plus transient-error retries) and return its text.

        A response cut off by the output limit returns whatever text was produced, so the
        caller's JSON validation records it as invalid, matching a truncated Ollama reply.
        """

        payload = self.request_payload(system=system, user=user, response_schema=response_schema)
        started = time.perf_counter()
        response, retries = self._with_retries("POST", "/responses", payload)
        elapsed = time.perf_counter() - started
        error = response.get("error")
        if isinstance(error, Mapping) and error:
            raise OpenAIResponseError(f"OpenAI response error: {error.get('message', error)}")
        status = response.get("status") if isinstance(response.get("status"), str) else None
        if status not in {None, "completed", "incomplete"}:
            raise OpenAIResponseError(f"OpenAI response status is {status!r}")
        usage = response.get("usage") if isinstance(response.get("usage"), Mapping) else None
        details = usage.get("output_tokens_details") if usage else None
        resolved = response.get("model")
        return OllamaChatResponse(
            content=_output_text(response),
            metadata=OpenAIGenerationMetadata(
                prompt_tokens=_optional_int(usage, "input_tokens"),
                completion_tokens=_optional_int(usage, "output_tokens"),
                total_duration_ns=None,
                load_duration_ns=None,
                prompt_eval_duration_ns=None,
                eval_duration_ns=None,
                wall_clock_seconds=elapsed,
                response_model=resolved if isinstance(resolved, str) else None,
                response_status=status,
                reasoning_tokens=_optional_int(details, "reasoning_tokens"),
                retries=retries,
            ),
        )
