"""
agent_controller.py
Utility wrapper that runs AdaptiveHybridEngine on a loop, schedules controller proposals,
logs history and supports checkpointing and replay.
"""

import time
import json
import os
import signal
from hybrid_relational_engine import AdaptiveHybridEngine
from core_er_epr import RelationalCore
import numpy as np
import logging

logger = logging.getLogger("agent_controller")
logging.basicConfig(level=logging.INFO)

class EngineRunner:
    def __init__(self, engine: AdaptiveHybridEngine, dt: float = 0.01, cycle_seconds: float = 1.0, llm_period: int = 10):
        self.engine = engine
        self.dt = dt
        self.cycle_seconds = cycle_seconds
        self.llm_period = llm_period
        self.keep_running = True
        self.iter = 0

    def signal_handler(self, sig, frame):
        logger.info("Signal received, stopping loop.")
        self.keep_running = False

    def run(self, max_iters: Optional[int] = None):
        signal.signal(signal.SIGINT, self.signal_handler)
        while self.keep_running:
            t0 = time.time()
            # advance physics
            diag = self.engine.step_physics(dt=self.dt, steps=1)
            # controller propose occasionally
            if self.iter % self.llm_period == 0:
                use_llm = (self.iter % (self.llm_period*5) == 0)
                res = self.engine.propose_and_apply(use_llm=use_llm)
                logger.info("Controller step iter=%d res=%s", self.iter, res)
            else:
                # small passive adapt by Hebbian already performed in step_physics
                pass
            self.iter += 1
            if max_iters is not None and self.iter >= max_iters:
                break
            dt = time.time() - t0
            sleep = max(0.0, self.cycle_seconds - dt)
            time.sleep(sleep)
        # on exit save history
        fname = f"engine_history_{int(time.time())}.json"
        with open(fname, "w") as f:
            json.dump(self.engine.history, f, indent=2, default=str)
        logger.info("Saved history to %s", fname)

def demo_run():
    # simple demo: random R
    R = (np.random.randn(4,4) + 1j*np.random.randn(4,4)) * 0.5
    core = RelationalCore(R)
    engine = AdaptiveHybridEngine(core)
    runner = EngineRunner(engine, dt=0.01, cycle_seconds=0.5, llm_period=6)
    runner.run(max_iters=60)

if __name__ == "__main__":
    demo_run()
