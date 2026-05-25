"""Configuration for supported translation models."""

from typing import TypedDict


class ModelInfo(TypedDict):
    """Information about a translation model."""
    id: str
    name: str
    provider: str
    description: str
    context_length: int


# Supported models via OpenAI Compatible API
SUPPORTED_MODELS: list[ModelInfo] = [
    {
        "id": "deepseek-v3.2",
        "name": "DeepSeek V3.2",
        "provider": "DeepSeek",
        "description": "DeepSeek 最新大模型",
        "context_length": 64000,
    },
    {
        "id": "deepseek-v4-flash",
        "name": "DeepSeek V4 Flash",
        "provider": "DeepSeek",
        "description": "DeepSeek V4 Flash，百万字超长上下文",
        "context_length": 1000000,
    },
    {
        "id": "qwen3-max",
        "name": "Qwen3 Max",
        "provider": "Alibaba",
        "description": "通义千问 3 旗舰版",
        "context_length": 32768,
    },
    {
        "id": "kimi-k2.5",
        "name": "Kimi K2.5",
        "provider": "Moonshot",
        "description": "Moonshot Kimi K2.5",
        "context_length": 128000,
    },
    {
        "id": "gpt-5-mini",
        "name": "GPT-5 Mini",
        "provider": "OpenAI",
        "description": "OpenAI GPT-5 Mini",
        "context_length": 128000,
    },
    {
        "id": "gpt-5.2",
        "name": "GPT-5.2",
        "provider": "OpenAI",
        "description": "OpenAI GPT-5.2",
        "context_length": 128000,
    },
    {
        "id": "gpt-4o",
        "name": "GPT-4o",
        "provider": "OpenAI",
        "description": "OpenAI GPT-4o",
        "context_length": 128000,
    },
    {
        "id": "gpt-4o-mini",
        "name": "GPT-4o Mini",
        "provider": "OpenAI",
        "description": "OpenAI GPT-4o Mini",
        "context_length": 128000,
    },
    {
        "id": "gemini-2.0-flash",
        "name": "Gemini 2.0 Flash",
        "provider": "Google",
        "description": "Google Gemini 2.0 Flash",
        "context_length": 128000,
    },
    {
        "id": "gemini-3.1-flash-lite",
        "name": "Gemini 3.1 Flash Lite",
        "provider": "Google",
        "description": "Google Gemini 3.1 Flash Lite",
        "context_length": 128000,
    },
    {
        "id": "gemini-3-flash-preview",
        "name": "Gemini 3 Flash Preview",
        "provider": "Google",
        "description": "Google Gemini 3 Flash Preview",
        "context_length": 128000,
    },
    {
        "id": "gemini-3-pro-preview",
        "name": "Gemini 3 Pro Preview",
        "provider": "Google",
        "description": "Google Gemini 3 Pro Preview",
        "context_length": 128000,
    },
    {
        "id": "claude-haiku-4-5",
        "name": "Claude Haiku 4.5",
        "provider": "Anthropic",
        "description": "Anthropic Claude Haiku 4.5",
        "context_length": 200000,
    },
    {
        "id": "claude-sonnet-4-5",
        "name": "Claude Sonnet 4.5",
        "provider": "Anthropic",
        "description": "Anthropic Claude Sonnet 4.5",
        "context_length": 200000,
    },
    {
        "id": "gpt-oss-120b",
        "name": "GPT OSS 120B",
        "provider": "OpenSource",
        "description": "Open Source GPT 120B",
        "context_length": 32768,
    },
    {
        "id": "DeepSeek-V3.2-Exp",
        "name": "DeepSeek V3.2 Exp",
        "provider": "DeepSeek",
        "description": "DeepSeek V3.2 Experimental",
        "context_length": 64000,
    },
    {
        "id": "qwen-plus-latest",
        "name": "Qwen Plus Latest",
        "provider": "Alibaba",
        "description": "Qwen Plus Latest Model",
        "context_length": 32768,
    },
]

# Default model if none specified
DEFAULT_MODEL = "deepseek-v3.2"


def get_model_by_id(model_id: str) -> ModelInfo | None:
    """Get model information by ID."""
    for model in SUPPORTED_MODELS:
        if model["id"] == model_id:
            return model
    return None


def is_model_supported(model_id: str) -> bool:
    """Check if a model is supported."""
    return get_model_by_id(model_id) is not None
