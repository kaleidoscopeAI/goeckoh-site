def __init__(self): self._subs: List[asyncio.Queue] = []
def subscribe(self): q = asyncio.Queue(maxsize=200); self._subs.append(q); return q
async def publish(self, msg: Dict[str, Any]):
    dead = []
    for q in self._subs:
        try: await q.put(msg)
        except asyncio.QueueFull: dead.append(q)
    if dead:
        for d in dead:
            try: self._subs.remove(d)
            except: pass

