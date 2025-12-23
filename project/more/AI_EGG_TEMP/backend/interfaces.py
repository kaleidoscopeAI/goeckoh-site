# backend/interfaces.py
from abc import ABC, abstractmethod
import numpy as np
from typing import List, Dict, Any

class BaseLLM(ABC):
    @abstractmethod
    async def reflect(self, prompt: str) -> str:
        pass

class BaseEmbedder(ABC):
    @abstractmethod
    def fit(self, texts: List[str]):
        pass
    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        pass

class BaseMemory(ABC):
    @abstractmethod
    def add(self, emb: np.ndarray, meta: Dict[str, Any]):
        pass
    @abstractmethod
    def query(self, emb: np.ndarray, k: int) -> list:
        pass

class BaseCrawler(ABC):
    @abstractmethod
    async def start(self, interval: float):
        pass
    @abstractmethod
    def stop(self):
        pass
