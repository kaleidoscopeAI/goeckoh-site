"""
hybrid_relational_engine.py
Adaptive hybrid engine tying RelationalCore (core_er_epr.py) to an adaptive controller and sandbox.

Main classes:
 - AdaptiveHybridEngine: manages a live RelationalCore, history, control loop
 - SandboxEvaluator: quick sandbox rollouts and reward computation
 - SimpleController: finite-difference numeric controller + evolutionary fallback
 - LLMAdapter (thin): parse-only LLM suggestions (optional Ollama)

Dependencies: numpy, requests (optional for LLM).
"""

from __future__ import annotations
import numpy as np
import copy
import time
import json
import math
import logging
from typing import Callable, Dict, Any, Optional, List, Tuple

from core_er_epr import RelationalCore, build_simple_hamiltonians, build_jump_ops_from_stress

logger = logging.getLogger("hybrid_engine")
logging.basicConfig(level=logging.INFO)

# -------------------------------
# Utilities
# -------------------------------
def ema_update(prev: float, value: float, alpha: float) -> float:
    return alpha * value + (1 - alpha) * prev

# -------------------------------
# Reward and diagnostics
# -------------------------------
class Diagnostics:
    def __init__(self, core: RelationalCore):
        self.core = core

    def compute(self) -> Dict[str, float]:
        rho = self.core.rho_S()
        purity = float(np.real(np.trace(rho @ rho)))
        entropy = self.core.entanglement_entropy()
        max_stress = float(np.max(np.abs(np.abs(self.core.R)**2 - np.abs(self.core.Q)**2)))
        avg_bridge = float(np.mean(self.core.bridge_strength_map(gamma=1.0)))
        probs = self.core.probs_born()
        # signal/noise as ratio of top row energy vs rest
        row_energy = np.sum(np.abs(self.core.R)**2, axis=1)
        snr = float((np.max(row_energy) + 1e-12) / (np.mean(row_energy) + 1e-12))
        return {'purity': purity, 'entropy': entropy, 'max_stress': max_stress, 'avg_bridge': avg_bridge, 'snr': snr, 'probs_sum': float(probs.sum())}

# -------------------------------
# Sandbox evaluator
# -------------------------------
class SandboxEvaluator:
    def __init__(self, steps: int = 8, dt: float = 0.01, reward_params: Optional[Dict[str,float]] = None):
        self.steps = steps
        self.dt = dt
        self.reward_params = reward_params or {'w_purity': 1.0, 'w_entropy': 0.5, 'w_stress': 0.2, 'w_snr': 0.5, 'w_phi': 0.0}

    def baseline_diag(self, core: RelationalCore, Hs: np.ndarray, Ha: np.ndarray, Hint: Optional[np.ndarray]) -> Dict[str,float]:
        diag = Diagnostics(core).compute()
        return diag

    def simulate_rollout(self, core: RelationalCore, Hs: np.ndarray, Ha: np.ndarray, Hint: Optional[np.ndarray], steps: Optional[int] = None) -> List[Dict[str,float]]:
        steps = steps or self.steps
        c = copy.deepcopy(core)
        traj = []
        for _ in range(steps):
            c.evolve_step(Hs, Ha, Hint, self.dt)
            c.adapt_bonds_hebb(eta=1e-5, decay=1e-6)
            diag = Diagnostics(c).compute()
            traj.append(diag)
        return traj

    def reward_from_traj(self, traj: List[Dict[str,float]], baselines: Dict[str,float]) -> float:
        # compute cumulative discounted reward (simple sum)
        w = self.reward_params
        total = 0.0
        gamma_r = 0.95
        for t, d in enumerate(traj):
            r_t = w['w_purity'] * (d['purity'] - baselines.get('purity',0.0)) \
                - w['w_entropy'] * (d['entropy'] - baselines.get('entropy',0.0)) \
                - w['w_stress'] * (d['max_stress'] - baselines.get('max_stress',0.0)) \
                + w['w_snr'] * (d['snr'] - baselines.get('snr',0.0))
            total += (gamma_r**t) * r_t
        return float(total)

    def evaluate_action(self, core: RelationalCore, action_fn: Callable[[RelationalCore], None], Hs: np.ndarray, Ha: np.ndarray, Hint: Optional[np.ndarray]) -> Dict[str, Any]:
        """
        Copies core, applies action_fn (mutates core copy), simulates rollout, returns reward & metrics.
        """
        c_copy = copy.deepcopy(core)
        try:
            action_fn(c_copy)
        except Exception as e:
            return {'ok': False, 'error': f"action failed: {e}", 'reward': -1e6}
        # baseline for comparison
        baseline_diag = Diagnostics(core).compute()
        traj = self.simulate_rollout(c_copy, Hs, Ha, Hint)
        reward = self.reward_from_traj(traj, baseline_diag)
        end_diag = traj[-1] if len(traj)>0 else baseline_diag
        ok = True
        # safety checks: no explosion
        if end_diag['purity'] < 1e-6 or end_diag['max_stress'] > 1e6:
            ok = False
        return {'ok': ok, 'reward': reward, 'end_diag': end_diag, 'traj': traj, 'baseline': baseline_diag}

# -------------------------------
# Simple numeric controller
# -------------------------------
class SimpleController:
    """
    Proposes small safe actions using finite-difference gradient approximation and evolutionary sampling fallback.
    Action space: {'mix_delta','gamma_delta','bond_scale_col':list or scalar}
    """
    def __init__(self, engine: 'AdaptiveHybridEngine', evaler: SandboxEvaluator, action_bounds: Optional[Dict[str,Tuple[float,float]]] = None):
        self.engine = engine
        self.evaler = evaler
        # bounds
        self.bounds = action_bounds or {'mix_delta':(-0.2,0.2), 'gamma_delta':(-0.5,0.5), 'bond_scale':(0.5,1.5)}
        self.finite_delta = 0.02
        self.step_scale = 0.1  # scale for gradient step
        self.pop_size = 12
        self.topk = 3

    def propose_gradient_action(self) -> Dict[str,float]:
        # baseline core
        core = self.engine.core
        Hs, Ha, Hint = self.engine.Hs, self.engine.Ha, self.engine.Hint
        baseline = self.evaler.simulate_rollout(core, Hs, Ha, Hint, steps=2)
        base_diag = baseline[-1] if baseline else Diagnostics(core).compute()
        base_reward = 0.0  # approximate baseline via reward_from_traj using an empty action
        base_reward = self.evaler.reward_from_traj(baseline, base_diag)
        # test each scalar action
        grads = {}
        # mix_delta
        def apply_mix_delta(c: RelationalCore, delta):
            c.mix = float(np.clip(c.mix + delta, 0.0, 1.0))
        # gamma
        def apply_gamma_delta(c: RelationalCore, delta):
            c.gamma = float(np.clip(c.gamma + delta, 0.0, 10.0))
        # bond_scale
        def apply_bond_scale(c: RelationalCore, scale):
            c.B = np.clip(c.B * scale, 0.0, 1e6)

        # finite difference for mix
        plus = self.evaler.evaluate_action(core, lambda x: apply_mix_delta(x, self.finite_delta), Hs, Ha, Hint)
        minus = self.evaler.evaluate_action(core, lambda x: apply_mix_delta(x, -self.finite_delta), Hs, Ha, Hint)
        g_mix = (plus['reward'] - minus['reward']) / (2*self.finite_delta)
        grads['mix_delta'] = g_mix

        plus = self.evaler.evaluate_action(core, lambda x: apply_gamma_delta(x, self.finite_delta), Hs, Ha, Hint)
        minus = self.evaler.evaluate_action(core, lambda x: apply_gamma_delta(x, -self.finite_delta), Hs, Ha, Hint)
        g_gamma = (plus['reward'] - minus['reward']) / (2*self.finite_delta)
        grads['gamma_delta'] = g_gamma

        plus = self.evaler.evaluate_action(core, lambda x: apply_bond_scale(x, 1.0 + self.finite_delta), Hs, Ha, Hint)
        minus = self.evaler.evaluate_action(core, lambda x: apply_bond_scale(x, 1.0 - self.finite_delta), Hs, Ha, Hint)
        # treat as derivative in scale space
        g_bond = (plus['reward'] - minus['reward']) / (2*self.finite_delta)
        grads['bond_scale'] = g_bond

        # normalize grads to propose step
        g_vec = np.array([grads['mix_delta'], grads['gamma_delta'], grads['bond_scale']])
        norm = np.linalg.norm(g_vec) + 1e-12
        step = (self.step_scale / norm) * g_vec
        action = {'mix_delta': float(np.clip(step[0], self.bounds['mix_delta'][0], self.bounds['mix_delta'][1])),
                  'gamma_delta': float(np.clip(step[1], self.bounds['gamma_delta'][0], self.bounds['gamma_delta'][1])),
                  'bond_scale': float(np.clip(1.0 + step[2], self.bounds['bond_scale'][0], self.bounds['bond_scale'][1]))}
        logger.info("Gradient propose grads=%s step=%s", grads, action)
        return action

    def propose_evolutionary(self) -> Dict[str,float]:
        # sample population around current params
        core = self.engine.core
        Hs, Ha, Hint = self.engine.Hs, self.engine.Ha, self.engine.Hint
        pop = []
        for i in range(self.pop_size):
            mix_delta = float(np.random.normal(loc=0.0, scale=0.05))
            gamma_delta = float(np.random.normal(loc=0.0, scale=0.1))
            bond_scale = float(np.random.normal(loc=1.0, scale=0.05))
            # clamp
            mix_delta = np.clip(mix_delta, self.bounds['mix_delta'][0], self.bounds['mix_delta'][1])
            gamma_delta = np.clip(gamma_delta, self.bounds['gamma_delta'][0], self.bounds['gamma_delta'][1])
            bond_scale = np.clip(bond_scale, self.bounds['bond_scale'][0], self.bounds['bond_scale'][1])
            cand = {'mix_delta':mix_delta, 'gamma_delta':gamma_delta, 'bond_scale':bond_scale}
            score = self.evaler.evaluate_action(core, lambda c, cand=cand: (c.mix := float(np.clip(c.mix + cand['mix_delta'],0,1))) , Hs, Ha, Hint)['reward']
            pop.append((score, cand))
        pop.sort(reverse=True, key=lambda x: x[0])
        top = [p[1] for p in pop[:self.topk]]
        # average top
        avg = {'mix_delta':np.mean([t['mix_delta'] for t in top]),
               'gamma_delta':np.mean([t['gamma_delta'] for t in top]),
               'bond_scale':np.mean([t['bond_scale'] for t in top])}
        logger.info("Evo top score: %s chosen avg action: %s", pop[0][0], avg)
        return avg

# -------------------------------
# LLM Adapter (thin)
# -------------------------------
class LLMAdapter:
    """
    Thin wrapper to ask LLM for action suggestions.
    It must return a JSON with allowed keys.
    We do no-code execution: parse-only.
    """
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "mistral", timeout: float = 6.0):
        self.base_url = base_url
        self.model = model
        self.timeout = timeout

    def suggest_action(self, diag: Dict[str,float], extra: str = "") -> Optional[Dict[str,float]]:
        prompt = f"""Diagnostics: {json.dumps(diag)}.
Return a JSON object with keys (mix_delta, gamma_delta, bond_scale) with small numbers.
Constrain mix_delta to [-0.2,0.2], gamma_delta [-0.5,0.5], bond_scale [0.5,1.5].
Respond ONLY with the JSON object."""
        if extra:
            prompt += "\n" + extra
        try:
            import requests
            resp = requests.post(f"{self.base_url}/api/generate", json={"model": self.model, "prompt": prompt, "stream": False, "options":{"temperature":0.0}}, timeout=self.timeout)
            if resp.status_code != 200:
                return None
            txt = resp.json().get("response","")
            # find first {...}
            s = txt.find("{"); e = txt.rfind("}")
            if s==-1 or e==-1:
                return None
            js = txt[s:e+1]
            obj = json.loads(js)
            # clamp
            obj2 = {}
            if 'mix_delta' in obj:
                obj2['mix_delta'] = float(np.clip(float(obj['mix_delta']), -0.2, 0.2))
            if 'gamma_delta' in obj:
                obj2['gamma_delta'] = float(np.clip(float(obj['gamma_delta']), -0.5, 0.5))
            if 'bond_scale' in obj:
                obj2['bond_scale'] = float(np.clip(float(obj['bond_scale']), 0.5, 1.5))
            return obj2
        except Exception as e:
            logger.warning("LLMAdapter failed: %s", e)
            return None

# -------------------------------
# Adaptive Hybrid Engine
# -------------------------------
class AdaptiveHybridEngine:
    def __init__(self, core: RelationalCore, Hs: Optional[np.ndarray]=None, Ha: Optional[np.ndarray]=None, Hint: Optional[np.ndarray]=None):
        self.core = core
        dS,dA = core.dims
        if Hs is None or Ha is None:
            self.Hs, self.Ha, self.Hint = build_simple_hamiltonians(dS, dA)
        else:
            self.Hs, self.Ha, self.Hint = Hs, Ha, Hint
        self.sandbox = SandboxEvaluator()
        self.controller = SimpleController(self, self.sandbox)
        self.llm = LLMAdapter()
        # history
        self.history = []
        # running baselines (EMA)
        diag = Diagnostics(self.core).compute()
        self.baselines = {k:diag[k] for k in diag}
        self.ema_alpha = 0.05
        # safety thresholds
        self.min_purity = 1e-6
        self.max_bond = 1e8

    def step_physics(self, dt: float = 0.01, steps: int = 1):
        for _ in range(steps):
            self.core.evolve_step(self.Hs, self.Ha, self.Hint, dt)
            self.core.adapt_bonds_hebb(eta=1e-5, decay=1e-6)
            diag = Diagnostics(self.core).compute()
            # update baselines
            for k in diag:
                self.baselines[k] = ema_update(self.baselines.get(k, diag[k]), diag[k], self.ema_alpha)
            # record
            self.history.append({'t':time.time(), 'diag': diag, 'mix': self.core.mix, 'gamma': self.core.gamma})
        return diag

    def propose_and_apply(self, use_llm: bool = False) -> Dict[str,Any]:
        # propose
        if use_llm:
            diag_now = Diagnostics(self.core).compute()
            suggestion = self.llm.suggest_action(diag_now)
            if suggestion is not None:
                # evaluate via sandbox
                def apply_fn(c):
                    c.mix = float(np.clip(c.mix + suggestion.get('mix_delta',0.0), 0.0,1.0))
                    c.gamma = float(np.clip(c.gamma + suggestion.get('gamma_delta',0.0), 0.0, 10.0))
                    c.B = np.clip(c.B * suggestion.get('bond_scale',1.0), 0.0, 1e6)
                res = self.sandbox.evaluate_action(self.core, apply_fn, self.Hs, self.Ha, self.Hint)
                if res['ok'] and res['reward'] > 0:
                    # commit
                    apply_fn(self.core)
                    logger.info("LLM action applied reward=%.6f", res['reward'])
                    return {'applied':True,'action':suggestion,'reward':res['reward']}
                else:
                    return {'applied':False,'action':suggestion,'reward':res.get('reward',None), 'ok':res['ok']}
        # numeric controller
        action = self.controller.propose_gradient_action()
        # sandbox test
        def apply_fn_core(c, action=action):
            c.mix = float(np.clip(c.mix + action.get('mix_delta',0.0), 0.0,1.0))
            c.gamma = float(np.clip(c.gamma + action.get('gamma_delta',0.0), 0.0, 10.0))
            c.B = np.clip(c.B * action.get('bond_scale',1.0), 0.0, 1e6)
        res = self.sandbox.evaluate_action(self.core, apply_fn_core, self.Hs, self.Ha, self.Hint)
        if res['ok'] and res['reward'] > 0:
            # commit real action
            apply_fn_core(self.core)
            logger.info("Numeric action applied reward=%.6f", res['reward'])
            return {'applied':True,'action':action,'reward':res['reward']}
        # fallback to evolutionary
        evo = self.controller.propose_evolutionary()
        def apply_fn_core2(c, action=evo):
            c.mix = float(np.clip(c.mix + action.get('mix_delta',0.0), 0.0,1.0))
            c.gamma = float(np.clip(c.gamma + action.get('gamma_delta',0.0), 0.0, 10.0))
            c.B = np.clip(c.B * action.get('bond_scale',1.0), 0.0, 1e6)
        res2 = self.sandbox.evaluate_action(self.core, apply_fn_core2, self.Hs, self.Ha, self.Hint)
        if res2['ok'] and res2['reward'] > 0:
            apply_fn_core2(self.core)
            logger.info("Evo action applied reward=%.6f", res2['reward'])
            return {'applied':True,'action':evo,'reward':res2['reward']}
        return {'applied':False, 'reason':'no positive action found', 'res':res, 'res2':res2}
