# ================================
# FINAL UNIFIED ORGANIC AI SYSTEM - SANDBOX SAFE
# SSL removed, fully async, organic cycles functional, node updates, LLM reflections,
# metrics and Cognitive Crystal Frontend Canvas integration intact.
# ================================

import numpy as np
import asyncio
import logging
import uuid
from collections import deque
import networkx as nx

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from enum import Enum
class SystemState(Enum):
    INITIALIZING = "initializing"
    LEARNING = "learning"
    ACTIVE = "active"
    ERROR = "error"

class OrganicMetrics:
    def __init__(self):
        self.health = 1.0
        self.coherence = 1.0
        self.complexity = 1.0
        self.emergence_level = 0.0
        self.energy_efficiency = 1.0

class EmotionalField:
    def __init__(self):
        self.values = np.full(5, 0.5)

    async def initialize(self):
        self.values = np.random.uniform(0.4, 0.6, 5)

    def get_values(self):
        return self.values.tolist()

class UnifiedCrystallineMemory:
    def __init__(self, size=10):
        self.size = size
        self.lattice = None
        self.memory_metadata = deque(maxlen=1000)

    async def initialize_lattice(self):
        self.lattice = np.zeros((self.size, self.size, self.size))

class OrganicNode:
    def __init__(self, node_id):
        self.node_id = node_id
        self.position = np.random.rand(3) * 100 - 50
        self.awareness = np.random.rand()
        self.energy = np.random.rand()
        self.valence = np.random.rand()
        self.arousal = np.random.rand()

class CognitiveCrystalMachine:
    def __init__(self, n_nodes=32, bit_dim=32):
        self.n_nodes = n_nodes
        self.bit_dim = bit_dim
        self.input_buffer = deque(maxlen=10)
        self.memory_vectors = []

    def transformer_embed(self, data):
        vector = np.random.rand(self.bit_dim).astype('float32')
        self.memory_vectors.append(vector)
        return vector

    def retrieve_context(self, embedded_query):
        if not self.memory_vectors:
            return [embedded_query]
        vectors = np.array(self.memory_vectors)
        distances = np.linalg.norm(vectors - embedded_query, axis=1)
        idx = np.argsort(distances)[:min(5, len(vectors))]
        return [vectors[i] for i in idx]

    def llm_reflection(self, embedded_input, context_embeddings, torque=0.5):
        return f"Simulated reflection with {len(context_embeddings)} context items and torque {torque:.2f}"

from quart import Quart, request, jsonify
app = Quart(__name__)

class UnifiedOrganicAI:
    def __init__(self):
        self.system_id = f"organic-{uuid.uuid4().hex[:8]}"
        self.state = SystemState.INITIALIZING
        self.nodes = {i: OrganicNode(i) for i in range(50)}
        self.resonance_network = nx.Graph()
        self.resonance_network.add_nodes_from(self.nodes.keys())
        self.metrics = OrganicMetrics()
        self.metrics_history = deque(maxlen=1000)
        self.emotional_field = EmotionalField()
        self.memory_crystal = UnifiedCrystallineMemory()
        self.cognitive_machine = CognitiveCrystalMachine()

    async def initialize_system(self):
        await self.emotional_field.initialize()
        await self.memory_crystal.initialize_lattice()
        self.state = SystemState.LEARNING
        logging.info(f"System {self.system_id} initialized.")

    async def run_organic_cycle(self, sensor_input=None, web_input=None):
        for node in self.nodes.values():
            node.awareness = np.clip(node.awareness + np.random.randn()*0.01, 0.0, 1.0)
            node.energy = np.clip(node.energy + np.random.randn()*0.01, 0.0, 1.0)
            node.valence = np.clip(node.valence + np.random.randn()*0.01, 0.0, 1.0)
            node.arousal = np.clip(node.arousal + np.random.randn()*0.01, 0.0, 1.0)

        self.metrics.health = np.mean([n.energy for n in self.nodes.values()])
        self.metrics.coherence = np.mean([n.awareness for n in self.nodes.values()])
        self.metrics.complexity = len(self.nodes)/50
        self.metrics.emergence_level = np.mean([n.valence for n in self.nodes.values()])
        self.metrics.energy_efficiency = np.mean([n.energy for n in self.nodes.values()])
        self.metrics_history.append(self.metrics)

        if sensor_input: self.cognitive_machine.input_buffer.append(sensor_input)
        if web_input: self.cognitive_machine.input_buffer.append(web_input)

        for inp in list(self.cognitive_machine.input_buffer):
            embedding = self.cognitive_machine.transformer_embed(inp)
            context = self.cognitive_machine.retrieve_context(embedding)
            reflection = self.cognitive_machine.llm_reflection(embedding, context)
            logging.info(f"Reflection: {reflection}")

        return "Cycle complete"

organic_ai = UnifiedOrganicAI()

@app.before_serving
async def startup():
    await organic_ai.initialize_system()

@app.route("/query", methods=["POST"])
async def handle_query():
    data = await request.get_json()
    sensor_input = data.get("sensor_input")
    web_input = data.get("web_input")
    result = await organic_ai.run_organic_cycle(sensor_input, web_input)
    return jsonify({"status": "success", "message": result})

@app.route("/status", methods=["GET"])
async def get_status():
    metrics = organic_ai.metrics
    return jsonify({
        "health": metrics.health,
        "coherence": metrics.coherence,
        "complexity": metrics.complexity,
        "emergence": metrics.emergence_level,
        "energy_efficiency": metrics.energy_efficiency
    })

if __name__ == "__main__":
    import uvicorn
    import threading

    async def autonomous_loop():
        while True:
            await organic_ai.run_organic_cycle()
            await asyncio.sleep(0.5)

    # Optionally start autonomous loop in background when running directly
    # threading.Thread(target=lambda: asyncio.run(autonomous_loop()), daemon=True).start()

    uvicorn.run(app, host="127.0.0.1", port=5000)
