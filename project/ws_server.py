import asyncio, json
class WSServer:
    def __init__(self, host, port, state_provider):
        self.host, self.port, self.state_provider = host, port, state_provider
    async def _handler(self, ws):
        while True:
            await asyncio.sleep(0.25)
            await ws.send(json.dumps(self.state_provider()))
    async def serve(self):
        import websockets
        async with websockets.serve(self._handler, self.host, self.port):
            print(f"🔌 WS server on ws://{self.host}:{self.port}")
            await asyncio.Future()
