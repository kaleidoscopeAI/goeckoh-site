from typing import Dict, Any
import asyncio
import time

class HardwareIntegrator:
    def __init__(self):
        self.connected_devices = []
        self._t0 = time.time()

    async def initialize(self) -> None:
        self.connected_devices = []
        self._t0 = time.time()

    async def sense_environment(self) -> Dict[str, Any]:
        await asyncio.sleep(0)
        uptime = time.time() - self._t0
        temp = 20.0 + ((uptime // 10) % 10) * 0.1
        light = float(((uptime // 60) % 2))
        return {'sensor_data': {'temp_c': temp, 'light': light}, 'device_states': {}, 'ts': time.time()}

    async def execute_controls(self, controls: Dict[str, Any]) -> None:
        await asyncio.sleep(0)

    async def shutdown(self) -> None:
        self.connected_devices = []
