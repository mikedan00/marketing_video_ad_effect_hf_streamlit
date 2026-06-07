import os
from dataclasses import dataclass
from typing import List

from dotenv import load_dotenv

load_dotenv()

try:
    import streamlit as st
except Exception:  # pragma: no cover
    st = None


def _secret_or_env(key: str, default=None):
    if st is not None:
        try:
            if key in st.secrets:
                return st.secrets[key]
        except Exception:
            pass
    return os.getenv(key, default)


def _as_int(value, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _as_float(value, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


@dataclass
class HFSettings:
    llm_engine: str
    hf_token: str
    router_model: str
    model_candidates: List[str]
    max_tokens: int
    temperature: float
    timeout_connect: int
    timeout_read: int
    max_retries: int

    @classmethod
    def load(cls) -> "HFSettings":
        candidates_raw = str(_secret_or_env(
            "HF_MODEL_CANDIDATES",
            "google/gemma-4-26B-A4B-it:deepinfra,google/gemma-4-26B-A4B-it:novita,google/gemma-4-31B-it:deepinfra,google/gemma-4-31B-it:together,Qwen/Qwen3.5-9B:together,Qwen/Qwen2.5-7B-Instruct:together",
        ))
        candidates = [x.strip() for x in candidates_raw.split(",") if x.strip()]
        router_model = str(_secret_or_env("HF_ROUTER_MODEL", candidates[0] if candidates else "google/gemma-4-26B-A4B-it:deepinfra"))
        if router_model not in candidates:
            candidates.insert(0, router_model)
        return cls(
            llm_engine=str(_secret_or_env("LLM_ENGINE", "hf_api")),
            hf_token=str(_secret_or_env("HF_TOKEN", "")),
            router_model=router_model,
            model_candidates=candidates,
            max_tokens=_as_int(_secret_or_env("HF_MAX_TOKENS", 1400), 1400),
            temperature=_as_float(_secret_or_env("HF_TEMPERATURE", 0.2), 0.2),
            timeout_connect=_as_int(_secret_or_env("HF_TIMEOUT_CONNECT", 10), 10),
            timeout_read=_as_int(_secret_or_env("HF_TIMEOUT_READ", 120), 120),
            max_retries=_as_int(_secret_or_env("HF_MAX_RETRIES", 3), 3),
        )
