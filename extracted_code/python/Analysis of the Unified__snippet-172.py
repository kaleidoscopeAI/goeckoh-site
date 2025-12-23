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

