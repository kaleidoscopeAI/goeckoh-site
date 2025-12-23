from __future__ import annotations

import hashlib
import re
import struct
from typing import Any

import ollama
import numpy as np

from core.settings import HeartSettings


class LocalLLM:
    """
    Thin wrapper around a local LLM backend.
    """

    def __init__(self, cfg: HeartSettings):
        self.cfg = cfg
        self.backend = cfg.llm_backend
        self.model = cfg.llm_model

    def generate(self, prompt: str, temperature: float, top_p: float) -> str:
        """
        Generate a response text from the local LLM.
        """
        temperature = max(0.1, float(temperature))
        top_p = float(np.clip(top_p, 0.1, 1.0))
        if self.backend == "ollama":
            res = ollama.generate(
                model=self.model,
                prompt=prompt,
                options={
                    "temperature": temperature,
                    "top_p": top_p,
                    "num_predict": self.cfg.llm_max_tokens,
                },
            )
            return self.enforce_first_person(res.get("response", "").strip())
        return self._fallback_response(prompt, temperature, top_p)

    def _fallback_response(self, prompt: str, temperature: float, top_p: float) -> str:
        """
        Safe, deterministic fallback if no LLM backend is available.
        """
        last_line = prompt.strip().splitlines()[-1] if prompt.strip() else ""
        if '"' in last_line:
            try:
                quoted = last_line.split('"')[-2]
            except Exception:
                quoted = last_line
        else:
            quoted = last_line
        return (
            f"I hear: {quoted}. I will answer slowly and calmly, "
            f"leaving space after each phrase so your thoughts can catch up."
        )

    def embed(self, text: str, dim: int | None = None) -> np.ndarray:
        """
        Get an embedding for a text string.
        """
        dim = dim or self.cfg.embedding_dim
        return self._hash_embedding(text, dim)

    @staticmethod
    def _hash_embedding(text: str, dim: int) -> np.ndarray:
        """
        Deterministic hash-based embedding.
        """
        vec = np.zeros(dim, dtype=np.float32)
        if not text:
            return vec
        tokens = text.lower().split()
        for tok in tokens:
            h = hashlib.sha256(tok.encode("utf-8")).digest()
            idx = struct.unpack("Q", h[:8])[0] % dim
            sign_raw = struct.unpack("I", h[8:12])[0]
            sign = 1.0 if (sign_raw % 2 == 0) else -1.0
            vec[idx] += sign
        norm = np.linalg.norm(vec) + 1e-8
        vec = vec / norm
        return vec

    @staticmethod
    def enforce_first_person(text: str) -> str:
        """
        Transform any second-person phrasing into first person as best we can.
        """
        _FIRST_PERSON_PATTERNS = [
            (r"\\byou are\\b", "I am"),
            (r"\\byou're\\b", "I'm"),
            (r"\\byou were\\b", "I was"),
            (r"\\byou'll\\b", "I'll"),
            (r"\\byou've\\b", "I've"),
            (r"\\byour\\b", "my"),
            (r"\\byours\\b", "mine"),
            (r"\\byou\\b", "I"),
        ]
        t = text
        for pattern, repl in _FIRST_PERSON_PATTERNS:
            t = re.sub(pattern, repl, t, flags=re.IGNORECASE)
        return t
