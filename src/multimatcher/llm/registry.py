from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Literal

Provider = Literal["openai", "openai_compat", "gemini", "anthropic"]

@dataclass(frozen=True)
class ModelSpec:
    alias: str                 # 사용자가 CLI로 넣는 이름 (ex. gpt-5, qwen3-max)
    provider: Provider         # openai | openai_compat | gemini | anthropic
    model: str                 # 실제 호출 문자열 (version)
    api_key_env: str           # 어떤 env var에서 키를 읽는지
    base_url: Optional[str] = None
    default_temperature: float = 0.0
    default_timeout_s: int = 60
    default_max_retries: int = 2


MODEL_REGISTRY: Dict[str, ModelSpec] = {
    # OpenAI
    "gpt-5": ModelSpec(
        alias="gpt-5",
        provider="openai",
        model="gpt-5-2025-08-07",
        api_key_env="OPENAI_API_KEY",
        default_temperature=1.0,
        default_max_retries=2,
    ),
    "gpt-5-mini": ModelSpec(
        alias="gpt-5-mini",
        provider="openai",
        model="gpt-5-mini-2025-08-07",
        api_key_env="OPENAI_API_KEY",
        default_temperature=1.0,
        default_max_retries=2,
    ),

    # OpenAI-compatible endpoints (Together)
    "gpt-oss-120b": ModelSpec(
        alias="gpt-oss-120b",
        provider="openai_compat",
        model="openai/gpt-oss-120b",
        api_key_env="OPENAI_API_KEY",
        base_url="https://api.together.xyz/v1",
        default_temperature=0.0,
        default_max_retries=4,
    ),
    "gpt-oss-20b": ModelSpec(
        alias="gpt-oss-20b",
        provider="openai_compat",
        model="openai/gpt-oss-20b",
        api_key_env="OPENAI_API_KEY",
        base_url="https://api.together.xyz/v1",
        default_temperature=0.0,
        default_max_retries=4,
    ),

    # Gemini
    "gemini-2.5-pro": ModelSpec(
        alias="gemini-2.5-pro",
        provider="gemini",
        model="gemini-2.5-pro-preview-06-05",
        api_key_env="GOOGLE_API_KEY",
        default_temperature=0.0,
        default_max_retries=2,
    ),
    "gemini-2.5-flash": ModelSpec(
        alias="gemini-2.5-flash",
        provider="gemini",
        model="gemini-2.5-flash-preview-05-20",
        api_key_env="GOOGLE_API_KEY",
        default_temperature=0.0,
        default_max_retries=2,
    ),

    # Claude
    "claude-sonnet-4.5": ModelSpec(
        alias="claude-sonnet-4.5",
        provider="anthropic",
        model="claude-sonnet-4-5-20250929",
        api_key_env="ANTHROPIC_API_KEY",
        default_temperature=0.0,
        default_timeout_s=60,
        default_max_retries=2,
    ),
    "claude-haiku-4.5": ModelSpec(
        alias="claude-haiku-4.5",
        provider="anthropic",
        model="claude-haiku-4-5-20251001",
        api_key_env="ANTHROPIC_API_KEY",
        default_temperature=0.0,
        default_timeout_s=60,
        default_max_retries=2,
    ),

    # Qwen via Novita (OpenAI-compatible)
    "qwen3-max": ModelSpec(
        alias="qwen3-max",
        provider="openai_compat",
        model="qwen/qwen3-max",
        api_key_env="OPENAI_API_KEY",
        base_url="https://api.novita.ai/v3/openai",
        default_temperature=1.0,
        default_max_retries=2,
    ),
    "qwen3-next-80b": ModelSpec(
        alias="qwen3-next-80b",
        provider="openai_compat",
        model="qwen/qwen3-next-80b-a3b-instruct",
        api_key_env="OPENAI_API_KEY",
        base_url="https://api.novita.ai/v3/openai",
        default_temperature=1.0,
        default_max_retries=2,
    ),
}


def get_model_spec(alias: str) -> ModelSpec:
    key = alias.strip().lower()
    if key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model alias: {alias}. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[key]