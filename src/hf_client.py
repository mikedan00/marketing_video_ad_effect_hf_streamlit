import json
import re
import time
from typing import Dict, Iterable, List, Optional, Tuple

import requests

HF_CHAT_COMPLETIONS_URL = "https://router.huggingface.co/v1/chat/completions"


class HFRouterError(RuntimeError):
    pass


def _extract_json(text: str) -> Dict:
    text = (text or "").strip()
    if not text:
        raise ValueError("LLM 응답이 비어 있습니다.")

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def chat_json(
    *,
    hf_token: str,
    model: str,
    messages: List[Dict],
    max_tokens: int = 1400,
    temperature: float = 0.2,
    timeout: Tuple[int, int] = (10, 120),
    max_retries: int = 3,
) -> Dict:
    if not hf_token or not hf_token.startswith("hf_"):
        raise HFRouterError("HF_TOKEN이 없거나 형식이 올바르지 않습니다. hf_로 시작하는 토큰을 설정하세요.")

    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "response_format": {"type": "json_object"},
        "stream": False,
    }

    last_error = None
    for attempt in range(1, int(max_retries) + 1):
        try:
            response = requests.post(HF_CHAT_COMPLETIONS_URL, headers=headers, json=payload, timeout=timeout)
            if response.status_code >= 400:
                raise HFRouterError(f"HTTP {response.status_code}: {response.text[:800]}")
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            result = _extract_json(content)
            result["_model_used"] = model
            return result
        except Exception as exc:
            last_error = exc
            if attempt < int(max_retries):
                time.sleep(min(2 ** attempt, 8))
            else:
                break

    raise HFRouterError(f"{model} 호출 실패: {last_error}")


def chat_json_with_fallback(
    *,
    hf_token: str,
    selected_model: str,
    model_candidates: Iterable[str],
    messages: List[Dict],
    max_tokens: int,
    temperature: float,
    timeout: Tuple[int, int],
    max_retries: int,
) -> Dict:
    ordered = []
    for m in [selected_model, *list(model_candidates)]:
        if m and m not in ordered:
            ordered.append(m)

    errors = []
    for model in ordered:
        try:
            return chat_json(
                hf_token=hf_token,
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=timeout,
                max_retries=max_retries,
            )
        except Exception as exc:
            errors.append(f"- {model}: {exc}")

    raise HFRouterError("모든 후보 모델 호출에 실패했습니다.\n" + "\n".join(errors))
