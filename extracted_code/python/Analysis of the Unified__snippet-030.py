# test_hybrid_engine.py
import numpy as np
from hybrid_relational_engine import AdaptiveHybridEngine
from core_er_epr import RelationalCore

def test_engine_basic_step():
    R = (np.random.randn(3,3) + 1j*np.random.randn(3,3)) * 0.5
    core = RelationalCore(R)
    engine = AdaptiveHybridEngine(core)
    diag = engine.step_physics(dt=0.01, steps=5)
    assert 'purity' in diag and 'entropy' in diag

def test_controller_propose():
    R = (np.random.randn(3,3) + 1j*np.random.randn(3,3)) * 0.5
    core = RelationalCore(R)
    engine = AdaptiveHybridEngine(core)
    res = engine.propose_and_apply(use_llm=False)
    assert isinstance(res, dict)
