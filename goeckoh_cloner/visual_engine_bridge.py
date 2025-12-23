import asyncio
import json
import threading
from typing import Dict, Optional

import numpy as np
import websockets

from config import WS_HOST, WS_PORT
from voice_profile import VoiceFingerprint


class BubbleBroadcaster:
    def __init__(self, host: str = WS_HOST, port: int = WS_PORT):
        self.host = host
        self.port = port
        self.clients = set()
        self._loop = asyncio.new_event_loop()
        self._server_started = threading.Event()
        th = threading.Thread(target=self._run_loop, daemon=True)
        th.start()
        self._server_started.wait()

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)

        async def handler(websocket, path):
            self.clients.add(websocket)
            try:
                async for _ in websocket:
                    # We don't consume incoming messages; we just keep the connection alive.
                    pass
            finally:
                self.clients.remove(websocket)

        server_coro = websockets.serve(handler, self.host, self.port)
        self._loop.run_until_complete(server_coro)
        self._server_started.set()
        print(f"[BubbleWS] WebSocket server on ws://{self.host}:{self.port}")
        self._loop.run_forever()

    async def _broadcast(self, payload: str):
        if not self.clients:
            return
        await asyncio.gather(*(ws.send(payload) for ws in self.clients))

    def send_state(self, state: Dict):
        if not self.clients:
            return
        payload = json.dumps(state)
        asyncio.run_coroutine_threadsafe(self._broadcast(payload), self._loop)


def compute_bubble_state(
    profile: VoiceFingerprint,
    control: Dict[str, np.ndarray],
    idx: int,
    idle: bool = False,
) -> Dict:
    """
    Map psychoacoustic controls to bubble + halo parameters for frame idx.

    Expects control dict with:
      "energy", "zcr", "tilt", "f0"
    """
    energy = float(control["energy"][idx])
    zcr = float(control["zcr"][idx])
    tilt = float(control["tilt"][idx])
    f0 = float(control["f0"][idx])

    # Core radius: personal space + loudness
    radius = profile.base_radius * (1.0 + 0.5 * energy)

    # Spike factor from ZCR ("Bouba/Kiki")
    spike = float(np.clip(zcr * 2.0, 0.0, 1.0))

    # Surface metallic vs matte from tilt
    metalness = float(np.clip(0.5 + tilt / 5.0, 0.0, 1.0))

    # Roughness from base_roughness modulated by loudness
    roughness = float(np.clip(profile.base_roughness + 0.3 * energy, 0.0, 1.0))

    # Hue from F0; map 80–400 Hz to [0, 1] (child's pitch sets the identity color)
    hue = float(np.clip((f0 - 80.0) / (400.0 - 80.0), 0.0, 1.0))

    # Halo / agency intensity: energy + spike, smoothed into [0, 1]
    halo_raw = 0.6 * energy + 0.4 * spike
    halo = float(np.clip(halo_raw, 0.0, 1.0))

    return {
        "radius": radius,
        "spike": spike,
        "metalness": metalness,
        "roughness": roughness,
        "hue": hue,
        "halo": halo,
        "idle": idle,
    }