class CrystallineHeart:
    def __init__(self, n_nodes=1024):
        self.n_nodes = n_nodes
        self.states = [random.uniform(-1, 1) for _ in range(n_nodes)]  # Internal states
        self.history = []  # For info term
        self.dt = 0.05
        self.lambda1, self.lambda2, self.lambda3 = 0.4, 0.3, 0.3  # Weights
        self.max_history = 10

    def entropy(self, values):
        # Shannon entropy sim (pure math)
        hist, _ = [], []
        for v in values:
            hist.append(math.exp(-v**2))  # Gaussian approx
        p = [h / sum(hist) for h in hist if sum(hist) > 0]
        return -sum(pi * math.log(pi + 1e-10) for pi in p) if p else 0.0

    def mutual_info(self, past, future):
        # Sim correlation as MI proxy
        if not past or not future:
            return 0.0
        return sum(p * f for p, f in zip(past, future)) / (len(past) + 1e-10)

    def step(self, arousal, agency_stress):
        stimulus = arousal + agency_stress

        # ODE update
        for i in range(self.n_nodes):
            drive = stimulus
            decay = -0.5 * self.states[i]
            coupling = 0.3 * (sum(random.uniform(-1, 1) * s for s in self.states) / self.n_nodes - self.states[i])  # Random B sim
            noise = random.gauss(0, 0.1)
            ds = drive + decay + coupling + noise
            self.states[i] += self.dt * ds

        # Life equation for GCL
        s_int = self.entropy(self.states)
        s_env = self.entropy([random.uniform(-1, 1) for _ in range(self.n_nodes)])  # Sim env
        ds_int = (s_int - self.entropy(self.history[-1] if self.history else self.states)) / self.dt if self.history else 0
        ds_env = random.gauss(0, 0.1)  # Sim env change
        term1 = self.lambda1 * (ds_int - ds_env)

        self.history.append(self.states[:])
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
        past = self.history[:-1] if len(self.history) > 1 else [self.states]
        future = [random.uniform(-1, 1) for _ in range(self.n_nodes)]  # Sim future env
        i_past_future = self.mutual_info([sum(p) for p in past], future)
        h_x = self.entropy(self.states)
        term2 = self.lambda2 * (i_past_future / (h_x + 1e-10))

        n_copies = self.n_nodes + random.randint(-1, 1)  # Sim growth
        dn = (n_copies - self.n_nodes) / self.dt
        term3 = self.lambda3 * (dn / (self.n_nodes + 1e-10))

        gcl = term1 + term2 + term3
        gcl = 1 / (1 + math.exp(-gcl))  # Sigmoid normalize 0-1

        return gcl

