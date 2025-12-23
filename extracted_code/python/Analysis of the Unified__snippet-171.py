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

