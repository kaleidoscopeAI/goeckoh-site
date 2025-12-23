async def viz_websocket_handler(websocket, path):
    async for message in websocket:
        snapshot = json.loads(message)
        await websocket.send(json.dumps({"nodes": snapshot}))

def start_viz_server():
    start_server = websockets.serve(viz_websocket_handler, "localhost", 8765)
    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()

