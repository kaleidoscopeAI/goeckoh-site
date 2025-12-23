from __future__ import annotations

import requests


class CCABridgeClient:
    """
    Lightweight HTTP bridge to the Cognitive Crystal AI (CCA) backend.

    Mirrors the API shape used in echo_prime.py:
    - POST /api/sensory/packet
    - GET  /api/commands/pull   (not yet consumed here)
    """

    def __init__(self, base_url: str, session_id: str, device_id: str):
        self.base_url = base_url.rstrip("/")
        self.session_id = session_id
        self.device_id = device_id

    def send_sensory_packet(self, packet: dict) -> None:
        url = f"{self.base_url}/api/sensory/packet"
        try:
            requests.post(url, json=packet, timeout=1.0)
        except Exception as e:
            print(f"[CCABridge] Failed to send packet: {e}")

    def pull_commands(self, since_ts: float) -> list[dict]:
        """
        Optional command pull helper. Echo's realtime loop does not yet
        apply remote commands directly, but the API is provided for
        future wiring or external tools.
        """
        url = f"{self.base_url}/api/commands/pull"
        try:
            r = requests.get(
                url,
                params={"session_id": self.session_id, "since": since_ts},
                timeout=1.0,
            )
            if r.status_code == 200 and r.content:
                return r.json()
        except Exception as e:
            print(f"[CCABridge] Failed to pull commands: {e}")
        return []

