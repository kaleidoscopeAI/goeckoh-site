class Planner:
    def __init__(self, hardware: SimulatedHardware, relational: RelationalMatrix):
        self.hw = hardware
        self.rel = relational
        # concrete actions map to methods to keep safe
        self.actions = [
            ("increase_freq", lambda: self.hw.set_frequency(self.hw.freq_GHz + 0.1)),
            ("decrease_freq", lambda: self.hw.set_frequency(self.hw.freq_GHz - 0.1)),
            ("toggle_gpio", lambda: self.hw.gpio.set_bit(random.randint(0, self.hw.gpio.size-1), random.randint(0,1))),
            ("no_op", lambda: None),
        ]

    def score_actions(self, thought: ThoughtEngines) -> List[Tuple[float, int]]:
        scores = []
        for idx, (name, fn) in enumerate(self.actions):
            sys_idx = idx % self.rel.n_system
            app_idx = idx % self.rel.n_apparatus
            rweight = abs(self.rel.bidirectional_weight(sys_idx, app_idx))
            cog_signal = float(np.tanh(np.mean(thought.b) + np.mean(thought.kappa)))
            score = rweight * (1.0 + cog_signal)
            # small diversity noise
            score += random.random() * 1e-6
            scores.append((score, idx))
        scores.sort(reverse=True, key=lambda x: x[0])
        return scores

    def select_and_execute(self, thought: ThoughtEngines) -> str:
        scored = self.score_actions(thought)
        weights = np.array([s for s, i in scored], dtype=float)
        if weights.sum() <= 0:
            idx = scored[0][1]
        else:
            probs = weights / weights.sum()
            choice = np.random.choice(len(scored), p=probs)
            idx = scored[choice][1]
        name, fn = self.actions[idx]
        fn()
        return name

