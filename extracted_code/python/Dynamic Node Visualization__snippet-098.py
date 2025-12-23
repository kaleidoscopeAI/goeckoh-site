class ParallelTempering:
    def __init__(self, evaluate_energy: Callable[[np.ndarray], float], temps: List[float]):
        self.eval = evaluate_energy
        self.temps = temps

    def run_ensemble(self, initial_states: List[np.ndarray], neighbor_fn: Callable[[np.ndarray, float], np.ndarray], steps_per_replica=200):
        n = len(initial_states)
        reps = [state.copy() for state in initial_states]
        energies = [self.eval(r) for r in reps]
        with ThreadPoolExecutor(max_workers=min(8, n)) as ex:
            futures = [ex.submit(self._local_search, reps[i], self.temps[i % len(self.temps)], neighbor_fn, steps_per_replica) for i in range(n)]
            results = [f.result() for f in futures]
        states = [r[0] for r in results]
        energies = [r[1] for r in results]
        for i in range(len(self.temps) - 1):
            e1 = energies[i]
            e2 = energies[i + 1]
            T1 = self.temps[i]
            T2 = self.temps[i + 1]
            delta = (e2 - e1) * (1.0 / T1 - 1.0 / T2)
            if delta < 0 or np.random.rand() < math.exp(-delta):
                states[i], states[i + 1] = states[i + 1], states[i]
                energies[i], energies[i + 1] = energies[i + 1], energies[i]
        best_idx = int(np.argmin(energies))
        return states[best_idx], energies[best_idx]

    def _local_search(self, state, temp, neighbor_fn, steps):
        cur = state.copy()
        cur_e = self.eval(cur)
        for _ in range(steps):
            cand = neighbor_fn(cur, temp)
            ce = self.eval(cand)
            delta = ce - cur_e
            if delta < 0 or np.random.rand() < math.exp(-delta / (temp + 1e-12)):
                cur = cand
                cur_e = ce
        return cur, cur_e

