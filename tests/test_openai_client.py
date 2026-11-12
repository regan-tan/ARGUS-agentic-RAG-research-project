"""OpenAI client tests. No test contacts the network: requests are injected fakes."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from typing import Any

import pytest

from argus.agents.argus import judge_json_schema, verifier_json_schema
from argus.agents.iterative import rewrite_json_schema
from argus.llm import (
    OllamaChatClient,
    OpenAIChatClient,
    OpenAIConfigurationError,
    OpenAIGenerationMetadata,
    OpenAIResponseError,
    OpenAITransportError,
    strict_json_schema,
)
from argus.llm.providers import (
    add_provider_arguments,
    build_chat_client,
    generator_identity_metadata,
    resolve_provider_arguments,
)
from argus.rag.standard import response_json_schema


def _response(text: str, *, status: str = "completed") -> dict[str, Any]:
    return {
        "model": "gpt-5.6-luna-2026-10-01",
        "status": status,
        "output": [
            {"type": "reasoning", "summary": []},
            {"type": "message", "content": [{"type": "output_text", "text": text}]},
        ],
        "usage": {
            "input_tokens": 120,
            "output_tokens": 30,
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    }


class FakeRequest:
    def __init__(self, *replies: object) -> None:
        self.replies = list(replies)
        self.calls: list[tuple[str, str, Mapping[str, object] | None]] = []

    def __call__(
        self, method: str, path: str, payload: Mapping[str, object] | None, timeout: float
    ) -> Mapping[str, Any]:
        self.calls.append((method, path, payload))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        assert isinstance(reply, Mapping)
        return reply


def _client(request: FakeRequest, **options: Any) -> tuple[OpenAIChatClient, list[float]]:
    sleeps: list[float] = []
    client = OpenAIChatClient(request=request, sleep=sleeps.append, api_key="test-key", **options)
    return client, sleeps


@pytest.mark.parametrize(
    "schema",
    [
        response_json_schema(),
        verifier_json_schema(),
        judge_json_schema(10, version="v4.1"),
        rewrite_json_schema(),
    ],
)
def test_every_argus_schema_converts_to_strict_mode(schema: dict[str, object]) -> None:
    strict = strict_json_schema(schema)

    def objects(node: object) -> list[Mapping[str, object]]:
        if isinstance(node, Mapping):
            found = [node] if node.get("type") == "object" else []
            return found + [item for value in node.values() for item in objects(value)]
        if isinstance(node, list):
            return [item for value in node for item in objects(value)]
        return []

    assert objects(strict)
    for node in objects(strict):
        assert node["additionalProperties"] is False
        assert set(node["required"]) == set(node["properties"])
    assert "additionalProperties" not in schema  # the original schema is not mutated


def test_strict_mode_rejects_optional_properties_instead_of_requiring_them() -> None:
    schema = {"type": "object", "required": ["a"], "properties": {"a": {}, "b": {}}}
    with pytest.raises(ValueError, match="optional properties"):
        strict_json_schema(schema)


def test_request_payload_mirrors_the_ollama_generation_settings() -> None:
    client, _ = _client(FakeRequest())
    schema = verifier_json_schema()
    payload = client.request_payload(system="SYS", user="USER", response_schema=schema)

    assert payload["model"] == "gpt-5.6-luna"
    assert payload["instructions"] == "SYS"
    assert payload["input"] == [{"role": "user", "content": "USER"}]
    assert payload["temperature"] == 0.0
    assert payload["max_output_tokens"] == 512
    assert payload["reasoning"] == {"effort": "none"}
    assert payload["store"] is False
    text_format = payload["text"]["format"]  # type: ignore[index]
    assert text_format["type"] == "json_schema"
    assert text_format["strict"] is True
    assert text_format["schema"] == strict_json_schema(schema)


def test_loose_mode_and_omitted_settings_send_the_schema_unchanged() -> None:
    client, _ = _client(FakeRequest(), schema_mode="loose", temperature=None, reasoning_effort=None)
    schema = rewrite_json_schema()
    payload = client.request_payload(system="S", user="U", response_schema=schema)

    assert "temperature" not in payload
    assert "reasoning" not in payload
    assert payload["text"]["format"]["strict"] is False  # type: ignore[index]
    assert payload["text"]["format"]["schema"] == schema  # type: ignore[index]


def test_chat_returns_output_text_and_usage_in_the_shared_response_shape() -> None:
    body = json.dumps({"label": "SUPPORT", "rationale": "Direct.", "sentence_indices": [1]})
    request = FakeRequest(_response(body))
    client, _ = _client(request)

    response = client.chat(system="S", user="U", response_schema=verifier_json_schema())

    assert response.content == body
    metadata = response.metadata
    assert isinstance(metadata, OpenAIGenerationMetadata)
    assert (metadata.prompt_tokens, metadata.completion_tokens) == (120, 30)
    assert metadata.response_model == "gpt-5.6-luna-2026-10-01"
    assert metadata.reasoning_tokens == 0
    assert metadata.retries == 0
    assert request.calls[0][:2] == ("POST", "/responses")
    assert {"prompt_tokens", "completion_tokens", "wall_clock_seconds"} <= set(metadata.to_dict())


def test_a_truncated_reply_is_returned_for_the_caller_to_reject_as_invalid_json() -> None:
    client, _ = _client(FakeRequest(_response('{"label": "SUPP', status="incomplete")))
    response = client.chat(system="S", user="U", response_schema=verifier_json_schema())
    assert response.content == '{"label": "SUPP'
    assert response.metadata.response_status == "incomplete"  # type: ignore[attr-defined]


def test_refusal_and_failed_status_raise_instead_of_inventing_output() -> None:
    refusal = {
        "status": "completed",
        "output": [{"type": "message", "content": [{"type": "refusal", "refusal": "No."}]}],
    }
    client, _ = _client(FakeRequest(refusal, {"status": "failed", "output": []}))
    with pytest.raises(OpenAIResponseError, match="refused"):
        client.chat(system="S", user="U", response_schema=rewrite_json_schema())
    with pytest.raises(OpenAIResponseError, match="failed"):
        client.chat(system="S", user="U", response_schema=rewrite_json_schema())


def test_rate_limits_are_retried_with_the_server_delay() -> None:
    limited = OpenAIResponseError("HTTP 429", status_code=429, retry_after=3.0)
    request = FakeRequest(limited, _response('{"query": "x"}'))
    client, sleeps = _client(request)

    response = client.chat(system="S", user="U", response_schema=rewrite_json_schema())

    assert response.metadata.retries == 1  # type: ignore[attr-defined]
    assert sleeps == [3.0]
    assert len(request.calls) == 2


def test_client_errors_are_not_retried() -> None:
    bad_request = OpenAIResponseError("HTTP 400", status_code=400)
    client, sleeps = _client(FakeRequest(bad_request))
    with pytest.raises(OpenAIResponseError, match="400"):
        client.chat(system="S", user="U", response_schema=rewrite_json_schema())
    assert sleeps == []


def test_transport_errors_stop_after_the_retry_budget() -> None:
    errors = [OpenAITransportError("down") for _ in range(3)]
    client, sleeps = _client(FakeRequest(*errors), max_retries=2, retry_base_seconds=1.0)
    with pytest.raises(OpenAITransportError):
        client.chat(system="S", user="U", response_schema=rewrite_json_schema())
    assert sleeps == [1.0, 2.0]


def test_missing_api_key_fails_preflight_without_a_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    request = FakeRequest()
    client = OpenAIChatClient(request=request)
    with pytest.raises(OpenAIConfigurationError, match="OPENAI_API_KEY"):
        client.model_identity()
    assert request.calls == []


def test_model_identity_records_settings_but_never_the_key() -> None:
    request = FakeRequest({"id": "gpt-5.6-luna", "owned_by": "openai", "created": 1})
    client, _ = _client(request)

    identity = client.model_identity()

    assert request.calls == [("GET", "/models/gpt-5.6-luna", None)]
    assert identity.digest is None
    recorded = identity.to_dict()
    assert recorded["provider"] == "openai"
    assert recorded["settings"]["reasoning_effort"] == "none"  # type: ignore[index]
    assert "test-key" not in json.dumps(recorded)


def _parse(*flags: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_provider_arguments(parser)
    args = parser.parse_args(list(flags))
    resolve_provider_arguments(args)
    return args


def test_ollama_remains_the_default_provider() -> None:
    args = _parse()
    client = build_chat_client(args, timeout_seconds=5.0, num_predict=512, num_ctx=8192, top_k=40)
    assert isinstance(client, OllamaChatClient)
    assert args.model == "qwen3:4b-instruct-2507-q4_K_M"
    assert generator_identity_metadata("ollama", {"digest": "d"}) == {"ollama": {"digest": "d"}}


def test_openai_provider_maps_the_shared_limits() -> None:
    args = _parse("--provider", "openai", "--omit-temperature", "--openai-schema", "loose")
    client = build_chat_client(args, timeout_seconds=5.0, num_predict=512, num_ctx=8192, top_k=40)
    assert isinstance(client, OpenAIChatClient)
    assert args.model == "gpt-5.6-luna"
    assert client.settings()["max_output_tokens"] == 512
    assert client.settings()["temperature"] is None
    assert client.settings()["schema_mode"] == "loose"
    assert generator_identity_metadata("openai", {"model": "m"}) == {
        "provider": "openai",
        "openai": {"model": "m"},
    }
