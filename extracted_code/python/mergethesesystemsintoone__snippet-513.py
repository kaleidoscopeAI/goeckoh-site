def __init__(self):
    self._subs: List[asyncio.Queue] = []

def subscribe(self):
    q = asyncio.Queue(maxsize=200)
    self._subs.append(q)
    return q

async def publish(self, msg: Dict[str, Any]):
    for q in list(self._subs):
        try:
            await q.put(msg)
        except asyncio.QueueFull:
            pass

