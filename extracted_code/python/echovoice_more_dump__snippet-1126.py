def __init__(
    self,
    node_id: Optional[str] = None,
    emotional_traits: Optional[Dict[str, float]] = None,
    learning_capacity: float = 0.1,
    position: Optional[List[float]] = None,
    seed: int = 0,
    replay_capacity: int = 2048,
):
    self.id = node_id or f"node-{uuid.uuid4().hex[:8]}"
    base_traits = {'valence_bias': 0.0, 'arousal_sensitivity': 0.5, 'coherence_preference': 0.5}
    self.emotional_traits = {**base_traits, **(emotional_traits or {})}
    self.learning_capacity = max(0.0, min(1.0, float(learning_capacity)))
    self.position = position or [0.0, 0.0, 0.0]
    self.is_healthy = True
    self._local_memory: List[Dict[str, Any]] = []
    self._predictor = LocalPredictor(alpha=0.2)
    self._rng = random.Random(int(seed))
    self._replay = ReplayBuffer(capacity=replay_capacity, rng=self._rng)

async def initialize(self) -> None:
    await asyncio.sleep(0)

async def sense_emotional_state(self) -> Dict[str, float]:
    valence = float(self.emotional_traits.get('valence_bias', 0.0) + (self._rng.random() - 0.5) * 0.02)
    arousal = float(
        max(0.0, min(1.0, self.emotional_traits.get('arousal_sensitivity', 0.5) + (self._rng.random() - 0.5) * 0.02))
    )
    coherence = float(
        max(0.0, min(1.0, self.emotional_traits.get('coherence_preference', 0.5) + (self._rng.random() - 0.5) * 0.02))
    )
    stance = 0.0
    return {'valence': valence, 'arousal': arousal, 'coherence': coherence, 'stance': stance}

async def integrate_conscious_insight(self, insight: Dict[str, Any]) -> Dict[str, Any]:
    mem = {'ts': asyncio.get_event_loop().time(), 'insight': insight}
    self._local_memory.append(mem)
    self._replay.append(mem)
    sig = float(len(str(insight)))
    est = self._predictor.update(sig)
    self.learning_capacity = max(0.0, min(1.0, self.learning_capacity + 0.0005))
    sample = self._replay.sample(8)
    consistency = 0.0
    if sample:
        lens = [len(str(s['insight'])) for s in sample]
        consistency = statistics.pvariance(lens) if len(lens) > 1 else 0.0
    return {'node_id': self.id, 'memory_id': len(self._local_memory) - 1, 'predictor_est': est, 'consistency': consistency}

async def get_local_memory_size(self) -> int:
    return len(self._local_memory)

async def get_emotional_state(self) -> Dict[str, float]:
    return await self.sense_emotional_state()

async def shutdown(self) -> None:
    self.is_healthy = False
    await asyncio.sleep(0)
