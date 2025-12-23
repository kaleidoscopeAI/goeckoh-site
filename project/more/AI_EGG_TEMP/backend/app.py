# backend/app.py
import asyncio
import logging
import yaml
from quart import Quart, websocket, request, jsonify

from factory import create_components
from cube_simulator import CubeSimulator
from transformation_engine import TransformationEngine

# --- Configuration and Global State ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

with open("config.yaml", 'r') as f:
    config = yaml.safe_load(f)

app = Quart(__name__)
clients = set()
bg_tasks = []

# --- Core AI System Class (Enhanced) ---
class UnifiedAISystem:
    def __init__(self, config):
        self.config = config
        self.sim = CubeSimulator(config['simulation'])
        
        # Factory creates components based on mode
        self.llm, self.crawler, self.memory, self.embedder = create_components(config, self)
        
        self.transform_engine = TransformationEngine(self.sim, self.memory)
        self.last_reflection = "System initialized."
        self.metrics = {} # Will be populated by the sim

    async def on_new_document(self, doc):
        """Callback for when crawler finds a new document."""
        emb = self.embedder.embed(doc['text'])
        context = self.memory.query(emb, k=3)
        context_text = ' | '.join([c[0].get('url', '') for c in context])
        prompt = f"New document ingested: {doc['url']}\nContext: {context_text}\nContentSnippet: {doc['text'][:200]}"
        
        reflection = await self.llm.reflect(prompt)
        self.last_reflection = reflection
        
        self.memory.add(emb, {'url': doc['url'], 'snippet': doc['text'][:200], 'reflection': reflection})
        
        await self.broadcast({
            'type': 'ingest',
            'url': doc['url'],
            'reflection': reflection,
        })

    async def run_cycle(self):
        """Main simulation and reflection loop."""
        events = self.sim.step()
        if events:
            self.transform_engine.handle_events(events)
        
        # Simplified trigger for reflection
        if self.sim.step_count % 20 == 0:
            # Create a prompt from current system state
            state_summary = f"Current state: Stress={self.sim.global_stress:.2f}, Harmony={self.sim.harmony:.2f}"
            self.last_reflection = await self.llm.reflect(state_summary)

        # Update metrics
        self.metrics = {
            "stress": self.sim.global_stress,
            "harmony": self.sim.harmony,
            "emergence": self.sim.emergence,
            "confidence": self.sim.confidence,
        }
        
        # Broadcast state to all clients
        state_data = {
            "type": "state_update",
            "nodes": [n.pos for n in self.sim.nodes.values()],
            "metrics": self.metrics,
            "thought": self.last_reflection
        }
        await self.broadcast(state_data)

    async def broadcast(self, message: dict):
        # (Same as in unified_embedded_system.py)
        pass

# --- Global AI Instance ---
ai_system = UnifiedAISystem(config)

# --- API Endpoints ---
@app.route('/api/status')
async def status():
    return jsonify(ai_system.metrics)

@app.route('/api/start_crawl', methods=['POST'])
async def start_crawl():
    task = asyncio.create_task(ai_system.crawler.start(interval=5.0))
    bg_tasks.append(task)
    return jsonify({'status': 'crawler_started'})

# ... (other endpoints like stop_crawl, query_reflect, etc.)

@app.websocket('/ws/updates')
async def ws_updates():
    # (Same as in unified_embedded_system.py)
    pass

async def main_loop():
    while True:
        await ai_system.run_cycle()
        await asyncio.sleep(ai_system.config['simulation']['time_step'])

if __name__ == '__main__':
    # Fit embedder if in offline mode
    if config['mode'] == 'offline':
        from components.offline.crawler import ensure_corpus
        corpus_path = './corpus'
        ensure_corpus(corpus_path)
        texts = []
        for f in os.listdir(corpus_path):
            if f.endswith('.txt'):
                with open(os.path.join(corpus_path, f), 'r', encoding='utf-8') as fh:
                    texts.append(fh.read())
        if texts:
            ai_system.embedder.fit(texts)

    loop = asyncio.get_event_loop()
    loop.create_task(main_loop())
    app.run(host='127.0.0.1', port=5001)
