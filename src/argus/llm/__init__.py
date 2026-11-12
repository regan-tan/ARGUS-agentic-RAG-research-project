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
from .openai_client import (
    DEFAULT_OPENAI_ENDPOINT,
    DEFAULT_OPENAI_MODEL,
    OpenAIChatClient,
    OpenAIConfigurationError,
    OpenAIError,
    OpenAIGenerationMetadata,
    OpenAIModelIdentity,
    OpenAIResponseError,
    OpenAITransportError,
    strict_json_schema,
)

__all__ = [
    "DEFAULT_OPENAI_ENDPOINT",
    "DEFAULT_OPENAI_MODEL",
    "OpenAIChatClient",
    "OpenAIConfigurationError",
    "OpenAIError",
    "OpenAIGenerationMetadata",
    "OpenAIModelIdentity",
    "OpenAIResponseError",
    "OpenAITransportError",
    "strict_json_schema",
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
