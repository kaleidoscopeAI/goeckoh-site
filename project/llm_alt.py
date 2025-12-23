"""Local LLM wrapper used by the Crystalline Heart."""
from __future__ import annotations

from typing import Any

import numpy as np

from .config import CompanionConfig
from .text_utils import enforce_first_person, hash_embedding

try:  # pragma: no cover - optional dependency
    import ollama  # type: ignore

    HAS_OLLAMA = True
except Exception:  # pragma: no cover - optional dependency
    ollama = None  # type: ignore
    HAS_OLLAMA = False


class LocalLLM:
    """Thin convenience wrapper for Ollama with safe fallback."""

    def __init__(self, config: CompanionConfig) -> None:
        self.cfg = config
        self.model = config.speech.llm_model
        self._has_ollama = HAS_OLLAMA

    def generate(self, prompt: str, temperature: float, top_p: float) -> str:
        temperature = max(0.1, float(temperature))
        top_p = float(np.clip(top_p, 0.1, 1.0))
        if self._has_ollama:
            try:
                res: dict[str, Any] = ollama.generate(  # type: ignore[misc]
                    model=self.model,
                    prompt=prompt,
                    options={
                        "temperature": temperature,
                        "top_p": top_p,
                        "num_predict": self.cfg.speech.llm_max_tokens,
                    },
                )
                raw = (res.get("response") or "").strip()
                return enforce_first_person(raw)
            except Exception as exc:  # pragma: no cover - runtime logging
                print(f"⚠️ Ollama error: {exc}")
        return self._fallback_response(prompt)

    def _fallback_response(self, prompt: str) -> str:
        lines = prompt.strip().splitlines()
        last_line = lines[-1] if lines else ""
        if '"' in last_line:
            try:
                quoted = last_line.split('"')[-2]
            except Exception:
                quoted = last_line
        else:
            quoted = last_line
        out = (
            f"I hear myself say: {quoted}. "
            "I speak slowly and calmly. I leave space after each phrase so my thoughts can catch up."
        )
        return enforce_first_person(out)

    def embed(self, text: str, dim: int) -> np.ndarray:
        return hash_embedding(text, dim)
