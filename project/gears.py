"""Functional gears message bus used by the Organic AI seed."""

from __future__ import annotations

import random
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass(frozen=True)
class Information:
    """Immutable data packet that flows between gears."""

    payload: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    source_gear: Optional[str] = None

    def new(self, payload: Any, source_gear: str) -> "Information":
        """Clone with a new payload/source while preserving metadata."""
        return Information(
            payload=payload,
            metadata=self.metadata.copy(),
            timestamp=time.time(),
            source_gear=source_gear,
        )


@dataclass(frozen=True)
class AudioData:
    """Payload for raw audio information."""

    waveform: np.ndarray
    sample_rate: int


@dataclass(frozen=True)
class SpeechData:
    """Payload for transcribed speech information."""

    raw_text: str
    corrected_text: str
    is_final: bool


@dataclass(frozen=True)
class EmotionData:
    """Payload for emotional state information from the Heart."""

    arousal: float
    valence: float
    coherence: float
    temperature: float
    raw_emotions: np.ndarray  # The full [N, 5] tensor


@dataclass(frozen=True)
class AgentDecision:
    """Payload for an AGI Seed's decision."""

    action: str  # e.g., "ECHO", "REGULATE", "LEARN"
    target_text: Optional[str] = None
    prosody_source: Optional[np.ndarray] = None
    mode: str = "inner"  # "inner", "outer", "coach"


class Message:
    def __init__(self, source: str, target: str, content: Any, energy: float = 1.0):
        self.source = source
        self.target = target
        self.content = content
        self.energy = energy
        self.timestamp = time.time()


class GearNode:
    def __init__(self, name: str, on_message=None):
        self.name = name
        self.on_message = on_message or (lambda m: None)
        self.queue = deque()
        self.connections: Dict[str, float] = {}
        self.energy = 1.0
        self.stress = 0.0
        self.running = False

    def connect(self, other: "GearNode", weight: float = 1.0) -> None:
        self.connections[other.name] = weight
        other.connections[self.name] = weight

    def receive(self, msg: Message) -> None:
        self.queue.append(msg)

    def send(self, target: str, content: Any, energy: float = 1.0) -> None:
        GearFabric.dispatch(Message(self.name, target, content, energy))

    def _adjust(self, msg: Message, latency: float) -> None:
        src = msg.source
        self.connections[src] = min(5.0, self.connections.get(src, 1.0) + 0.05)
        for peer in list(self.connections.keys()):
            if peer != src:
                self.connections[peer] = max(0.1, self.connections[peer] * 0.999)
        self.stress = min(
            1.0, 0.8 * self.stress + 0.2 * (latency + min(1.0, len(self.queue) / 10.0))
        )
        self.energy = max(0.1, min(2.0, self.energy + 0.02 - self.stress * 0.05))
        if self.stress > 0.7:
            time.sleep(self.stress * 0.1)

    def _recover(self) -> None:
        self.stress = max(0.0, self.stress - 0.01)
        self.energy = min(2.0, self.energy + 0.005)

    def start(self) -> None:
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self) -> None:
        self.running = False

    def _loop(self) -> None:
        while self.running:
            if not self.queue:
                self._recover()
                time.sleep(0.05)
                continue
            msg = self.queue.popleft()
            t0 = time.time()
            result = self.on_message(msg)
            latency = time.time() - t0
            self._adjust(msg, latency)
            if result is not None:
                GearFabric.dispatch(
                    Message(self.name, msg.source, result, msg.energy * 0.9)
                )


class GearFabric:
    nodes: Dict[str, GearNode] = {}

    @classmethod
    def register(cls, node: GearNode) -> None:
        cls.nodes[node.name] = node

    @classmethod
    def dispatch(cls, msg: Message) -> None:
        if msg.target and msg.target in cls.nodes:
            cls.nodes[msg.target].receive(msg)
            return

        sender = cls.nodes.get(msg.source)
        if not sender:
            return

        for peer, weight in sender.connections.items():
            peer_node = cls.nodes.get(peer)
            if not peer_node:
                continue
            throttle = max(0.1, 1.0 - peer_node.stress)
            if random.random() < (weight / 5.0) * throttle:
                peer_node.receive(msg)

    @classmethod
    def snapshot(cls) -> Dict[str, Dict[str, Any]]:
        return {
            name: {
                "energy": round(node.energy, 3),
                "stress": round(node.stress, 3),
                "connections": dict(node.connections),
            }
            for name, node in cls.nodes.items()
        }
