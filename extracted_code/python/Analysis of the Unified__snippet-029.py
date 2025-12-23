"""
orchestrator.py
Lightweight job manager around AdaptiveHybridEngine enabling multiple concurrent jobs (single-process),
sandbox, apply_action API, and audit logging.
"""

import uuid
import json
import time
from typing import Dict, Any
from hybrid_relational_engine import AdaptiveHybridEngine, SandboxEvaluator
from core_er_epr import RelationalCore
import numpy as np
import logging

logger = logging.getLogger("orchestrator")
logging.basicConfig(level=logging.INFO)

class Orchestrator:
    def __init__(self):
        self.jobs: Dict[str, Dict[str,Any]] = {}
        self.audit = []

    def start_job(self, R: np.ndarray, policy: Dict[str,Any] = None) -> str:
        core = RelationalCore(R)
        engine = AdaptiveHybridEngine(core)
        job_id = f"job_{uuid.uuid4().hex[:8]}"
        self.jobs[job_id] = {'engine': engine, 'policy':policy or {}, 't':time.time()}
        logger.info("Started job %s dims=%s", job_id, core.dims)
        return job_id

    def step(self, job_id: str, steps:int = 1):
        rec = self.jobs[job_id]
        eng: AdaptiveHybridEngine = rec['engine']
        for _ in range(steps):
            eng.step_physics()
        return eng.history[-1] if eng.history else None

    def propose_and_apply(self, job_id: str, use_llm: bool = False):
        rec = self.jobs[job_id]
        eng: AdaptiveHybridEngine = rec['engine']
        res = eng.propose_and_apply(use_llm=use_llm)
        # audit
        self.audit.append({'job':job_id, 't':time.time(), 'action':res})
        return res

    def snapshot(self, job_id: str):
        rec = self.jobs[job_id]
        eng: AdaptiveHybridEngine = rec['engine']
        return {'diag': eng.step_physics(), 'mix':eng.core.mix, 'gamma':eng.core.gamma}

    def save_audit(self, fname: str = "audit_log.json"):
        with open(fname, "w") as f:
            json.dump(self.audit, f, indent=2, default=str)
