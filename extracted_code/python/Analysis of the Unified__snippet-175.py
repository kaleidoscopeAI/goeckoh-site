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
