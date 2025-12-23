"""
Simplified Crystalline Memory for Echo integration.

You can later swap this implementation with your full FAISS / lattice-based
memory while keeping the same interface.
"""

from __future__ import annotations

from typing import Any, Dict, List


class CrystallineMemory:
    def __init__(self) -> None:
        # Just a chronological log of packets for now
        self._log: List[Dict[str, Any]] = []

    def store_packet(self, packet: Dict[str, Any]) -> None:
        self._log.append(packet)

    def recent_context(self, n: int = 8) -> List[Dict[str, Any]]:
        if n <= 0:
            return []
        return self._log[-n:]
