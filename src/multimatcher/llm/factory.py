from __future__ import annotations

import os
from typing import Optional, Any

from multimatcher.llm.registry import ModelSpec

def _require_env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing environment variable: {name}")
    return v


def build_chat_model(
    spec: ModelSpec,
    temperature: Optional[float] = None,
    timeout_s: Optional[int] = None,
    max_retries: Optional[int] = None,
) -> Any:
    """
    Returns a LangChain chat model instance.
    Type is Any to avoid importing langchain base classes everywhere.
    """
    temp = float(spec.default_temperature if temperature is None else temperature)
    tmo = int(spec.default_timeout_s if timeout_s is None else timeout_s)
    retries = int(spec.default_max_retries if max_retries is None else max_retries)

    api_key = _require_env(spec.api_key_env)

    if spec.provider in ("openai", "openai_compat"):
        from langchain_openai import ChatOpenAI

        kwargs = dict(
            model=spec.model,
            temperature=temp,
            max_tokens=None,
            timeout=tmo,
            max_retries=retries,
        )
        # openai_compat이면 base_url과 api_key를 명시
        if spec.provider == "openai_compat":
            kwargs["api_key"] = api_key
            if spec.base_url:
                kwargs["base_url"] = spec.base_url
        # openai(provider="openai")는 보통 OPENAI_API_KEY만 있으면 동작하지만,
        # 명시해도 문제없으니 통일하고 싶으면 kwargs["api_key"]=api_key 넣어도 됨.
        return ChatOpenAI(**kwargs)

    if spec.provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        # Gemini는 GOOGLE_API_KEY를 env로 읽는다(라이브러리가 자동 사용)
        return ChatGoogleGenerativeAI(
            model=spec.model,
            temperature=temp,
            max_output_tokens=None,
        )

    if spec.provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=spec.model,
            temperature=temp,
            max_tokens=None,
            timeout=tmo,
            api_key=api_key,
        )

    raise ValueError(f"Unsupported provider: {spec.provider}")