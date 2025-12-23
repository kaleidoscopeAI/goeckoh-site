def __init__(self, device='cpu'):
    self.device = device
    self.embedder = LocalEmbedder(n_components=64)
    self.llm = EmbeddedLLM(device=device)
    self.memory = CrystallineMemory(dim=64, capacity=2048)
    self.nodes = {}
    self.metrics = {'energy_efficiency': 1.0, 'coherence': 0.5}
    self.knowledge_points = []
    self.listeners = []  # websocket listeners
    # initialize simple nodes
    for i in range(16):
        self.nodes[i] = {'awareness': random.random(), 'energy': random.random(), 'pos': np.random.rand(3).tolist()}

async def on_new_document(self, doc: Document):
    # Query nearest memories
    q = self.memory.query(doc.embedding, k=3)
    # Compose a local context
    context_texts = ' | '.join([m.get('url','') for m, s in q])
    prompt = f"New document ingested: {os.path.basename(doc.url)}\nContext: {context_texts}\nContentSnippet: {doc.text[:200]}"
    reflection = self.llm.reflect(prompt)
    # update memory and knowledge points
    self.memory.add(doc.embedding, {'url': doc.url, 'snippet': doc.text[:200], 'reflection': reflection})
    self.knowledge_points.append({'url': doc.url, 'coords': doc.embedding[:3].tolist(), 'score': doc.score})
    # adjust nodes slightly based on doc
    for n in self.nodes.values():
        n['awareness'] = min(1.0, n['awareness'] + 0.01 * doc.score)
        n['energy'] = max(0.0, n['energy'] - 0.005 * doc.score)
    # broadcast to listeners
    await self.broadcast({
        'type': 'ingest',
        'url': doc.url,
        'reflection': reflection,
        'score': doc.score,
        'timestamp': doc.timestamp
    })

async def broadcast(self, message: Dict[str, Any]):
    # send to websocket listeners
    to_remove = []
    for ws in list(self.listeners):
        try:
            await ws.send(json.dumps(message))
        except Exception:
            to_remove.append(ws)
    for r in to_remove:
        try: self.listeners.remove(r)
        except: pass

def register_ws(self, ws):
    self.listeners.append(ws)

