"""
LocalLLMCore: embedded open-source LLM (no external APIs).

Uses llama.cpp Python bindings to load a GGUF model from disk and
provides a simple chat() method for other modules (Echo, AGI, etc.).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any

from llama_cpp import Llama  # pip install llama-cpp-python


@dataclass
class LocalLLMConfig:
    model_path: Path
    n_ctx: int = 4096
    n_threads: int = 8
    n_gpu_layers: int = 0  # 0 = CPU only; adjust if GPU is available
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 256


class LocalLLMCore:
    """
    Singleton-style LLM runtime, thread-safe, used everywhere.

    All high-level reasoning (Echo planning, ABA wording, AGI thought)
    should call this instead of shelling out to external APIs.
    """

    _instance: "LocalLLMCore | None" = None
    _lock = threading.Lock()

    def __init__(self, cfg: LocalLLMConfig):
        self.cfg = cfg
        if not cfg.model_path.exists():
            raise FileNotFoundError(
                f"Local LLM model not found at {cfg.model_path}. "
                "Place a GGUF model file there (e.g. echo_brain.Q4_K_M.gguf)."
            )

        self._llm = Llama(
            model_path=str(cfg.model_path),
            n_ctx=cfg.n_ctx,
            n_threads=cfg.n_threads,
            n_gpu_layers=cfg.n_gpu_layers,
            logits_all=False,
            vocab_only=False,
        )
        self._call_lock = threading.Lock()

    @classmethod
    def init_global(cls, cfg: LocalLLMConfig) -> "LocalLLMCore":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(cfg)
            return cls._instance

    @classmethod
    def get_global(cls) -> "LocalLLMCore":
        if cls._instance is None:
            raise RuntimeError("LocalLLMCore not initialized. Call init_global() first.")
        return cls._instance

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        stop: List[str] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """
        Simple chat-style call.

        We manually format system + user messages into a single prompt for
        instruct-style models (e.g. Llama 3, Mistral, etc.).
        """
        if stop is None:
            stop = ["</s>"]
        if temperature is None:
            temperature = self.cfg.temperature
        if max_tokens is None:
            max_tokens = self.cfg.max_tokens

        prompt = (
            f"<<SYS>>{system_prompt.strip()}<</SYS>>\n\n"
            f"<<USER>>{user_prompt.strip()}<</USER>>\n\n"
            f"<<ASSISTANT>>"
        )

        with self._call_lock:
            out: Dict[str, Any] = self._llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=self.cfg.top_p,
                stop=stop,
            )

        text = out["choices"][0]["text"]
        return text.strip()