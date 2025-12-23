import asyncio
import logging
import uuid
from typing import Dict, Any, List
from ..core.types import OrganicMetrics, SystemState
from ..utils.logger import configure_logging
from ..nodes.node import OrganicNode
from ..memory.crystal import UnifiedCrystallineMemory
from ..optimizer.quantum_like import QuantumInspiredOptimizer
from ..consciousness.engine import ConsciousnessEngine
from ..collective.engine import CollectiveIntelligence
from ..evolution.manager import EvolutionManager
from ..hardware.integrator import HardwareIntegrator

class UnifiedOrganicAI:
    def __init__(self, config: Dict[str, Any] = None):
        configure_logging()
        self.logger = logging.getLogger('UnifiedOrganicAI')
        self.system_id = f"organic-ai-{uuid.uuid4().hex[:8]}"
        self.state = SystemState.INITIALIZING
        self.config = config or {}
        self.nodes: Dict[str, OrganicNode] = {}
        self.memory = UnifiedCrystallineMemory(seed=self.config.get('seed', 0))
        self.optimizer = QuantumInspiredOptimizer(seed=self.config.get('seed', 0))
        self.consciousness = ConsciousnessEngine()
        self.collective = CollectiveIntelligence()
        self.evolution = EvolutionManager(seed=self.config.get('seed', 0))
        self.hardware = HardwareIntegrator()
        self.metrics = OrganicMetrics()
        self.metrics_history: List[Dict[str, Any]] = []
        self._stop_requested = False
        self._node_semaphore = asyncio.Semaphore(self.config.get('max_concurrent_nodes', 64))

    async def initialize_system(self) -> None:
        self.logger.info('Initializing system')
        await self.memory.initialize_lattice()
        await self.consciousness.initialize()
        await self.hardware.initialize()
        initial = int(self.config.get('initial_nodes', 6))
        for i in range(initial):
            traits = {'valence_bias': (i % 3 - 1) * 0.1, 'arousal_sensitivity': 0.5, 'coherence_preference': 0.5}
            node = OrganicNode(emotional_traits=traits, seed=self.config.get('seed', 0) + i)
            await node.initialize()
            self.nodes[node.id] = node
        self.state = SystemState.LEARNING
        self.logger.info('Initialization complete')

    async def _gather_node_emotions(self) -> List[Dict[str, float]]:
        async def wrap(node: OrganicNode) -> Dict[str, float]:
            async with self._node_semaphore:
                return await node.sense_emotional_state()
        coros = [wrap(n) for n in self.nodes.values() if n.is_healthy]
        results = []
        if coros:
            wrapped = [asyncio.wait_for(c, timeout=self.config.get('node_timeout', 0.5)) for c in coros]
            results = await asyncio.gather(*wrapped, return_exceptions=True)
        clean = [r for r in results if not isinstance(r, Exception) and isinstance(r, dict)]
        return clean

    async def run_organic_cycle(self) -> None:
        cycle_start = asyncio.get_event_loop().time()
        try:
            hw = await asyncio.wait_for(self.hardware.sense_environment(), timeout=self.config.get('hw_timeout', 1.0))
            node_emotions = await self._gather_node_emotions()
            emotional_blob = {
                'hardware': hw,
                'emotional_field': {
                    'valence': self._avg([e.get('valence', 0.0) for e in node_emotions]) if node_emotions else 0.0,
                    'coherence': self._avg([e.get('coherence', 0.0) for e in node_emotions]) if node_emotions else 0.0,
                },
                'external': {},
            }
            crystals = await self.memory.form_crystals(emotional_blob)
            annealed = await self.memory.anneal_structure(crystals)
            s = await self.optimizer.create_superposition({'annealed': annealed})
            s['annealed'] = annealed
            s = await self.optimizer.apply_entanglement(s)
            opt = await self.optimizer.optimize_with_interference(s)
            final_decision = await self.optimizer.collapse_superposition(opt)
            self_awareness = await self.consciousness.update_self_model(final_decision)
            intentions = await self.consciousness.form_intentions(self_awareness)
            reflection = await self.consciousness.reflect_on_cognition(intentions)
            qualia = await self.consciousness.simulate_qualia(reflection)
            async def node_integrate(node: OrganicNode) -> Dict[str, Any]:
                async with self._node_semaphore:
                    return await node.integrate_conscious_insight({'self_awareness': self_awareness, 'intentions': intentions})
            coros = [node_integrate(n) for n in self.nodes.values() if n.is_healthy]
            node_responses = []
            if coros:
                wrapped = [asyncio.wait_for(c, timeout=self.config.get('node_timeout', 0.5)) for c in coros]
                node_responses = await asyncio.gather(*wrapped, return_exceptions=True)
            node_responses = [r for r in node_responses if not isinstance(r, Exception) and r]
            emergence = await self.collective.detect_emergence(node_responses)
            collective_decision = await self.collective.make_collective_decision(node_responses, emergence)
            if self._is_action_safe(collective_decision):
                await asyncio.wait_for(self.hardware.execute_controls({'decision': collective_decision}), timeout=self.config.get('hw_timeout', 1.0))
            if self._should_evolve():
                await self.evolution.evolve_population(self.nodes)
            await self._update_metrics(cycle_start)
        except asyncio.TimeoutError:
            self.logger.warning('Timeout during cycle')
        except Exception as e:
            self.logger.exception('Error in organic cycle: %s', e)

    def _is_action_safe(self, action: Dict[str, Any]) -> bool:
        a = action.get('collective_action')
        return a in (None, 'noop', 'coordinated', 'coordinated_action')

    def _avg(self, arr: List[float]) -> float:
        if not arr:
            return 0.0
        return sum(arr) / len(arr)

    def _should_evolve(self) -> bool:
        return False

    async def _update_metrics(self, cycle_start_time: float) -> None:
        healthy = sum(1 for n in self.nodes.values() if n.is_healthy)
        total = len(self.nodes)
        self.metrics.health = healthy / total if total else 0.0
        self.metrics_history.append({'ts': asyncio.get_event_loop().time(), 'metrics': self.metrics})

    async def run_continuous(self, cycles: int = None) -> None:
        await self.initialize_system()
        i = 0
        while not self._stop_requested and (cycles is None or i < cycles):
            await self.run_organic_cycle()
            i += 1
            await asyncio.sleep(self.config.get('cycle_interval', 0.1))
        await self.graceful_shutdown()

    async def request_stop(self) -> None:
        self._stop_requested = True

    async def graceful_shutdown(self) -> None:
        self.logger.info('Graceful shutdown started')
        tasks = [n.shutdown() for n in self.nodes.values()]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        await self.hardware.shutdown()
        self.logger.info('Shutdown complete')

    async def save_snapshot(self, path: str) -> None:
        await self.memory.snapshot(path)

    async def restore_snapshot(self, path: str) -> None:
        await self.memory.restore(path)

    async def get_system_report(self) -> Dict[str, Any]:
        return {'system_id': self.system_id, 'state': self.state.value, 'node_count': len(self.nodes), 'metrics': self.metrics.__dict__, 'memory_crystal_count': await self.memory.get_crystal_count()}
