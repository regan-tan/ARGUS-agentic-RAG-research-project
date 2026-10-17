"""Small local-model clients used by ARGUS experiments."""

from .ollama import (
    DEFAULT_OLLAMA_ENDPOINT,
    DEFAULT_QWEN_MODEL,
    GenerationMetadata,
    OllamaChatClient,
    OllamaChatResponse,
    OllamaError,
    OllamaMissingModelError,
    OllamaModelIdentity,
    OllamaResponseError,
    OllamaTransportError,
)

__all__ = [
    "DEFAULT_OLLAMA_ENDPOINT",
    "DEFAULT_QWEN_MODEL",
    "GenerationMetadata",
    "OllamaChatClient",
    "OllamaChatResponse",
    "OllamaError",
    "OllamaMissingModelError",
    "OllamaModelIdentity",
    "OllamaResponseError",
    "OllamaTransportError",
]
