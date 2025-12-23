def __init__(self, evaluate_energy: Callable[[np.ndarray], float], initial_temp=1.0, final_temp=1e-3, schedule='exp', steps=2000):
    self.eval = evaluate_energy
    self.initial_temp = initial_temp
    self.final_temp = final_temp
    self.schedule = schedule
    self.steps = steps

def temp_at(self, step):
    if self.schedule == 'exp':
        return self.initial_temp * (self.final_temp / self.initial_temp) ** (step / max(1, self.steps - 1))
    elif self.schedule == 'linear':
        return self.initial_temp + (self.final_temp - self.initial_temp) * (step / max(1, self.steps - 1))
    return self.initial_temp

def anneal(self, initial_state: np.ndarray, neighbor_fn: Callable[[np.ndarray, float], np.ndarray]):
    state = initial_state.copy()
    best = state.copy()
    best_e = self.eval(state)
    current_e = best_e
    for step in range(self.steps):
        T = self.temp_at(step)
        candidate = neighbor_fn(state, T)
        ce = self.eval(candidate)
        delta = ce - current_e
        accept = delta < 0 or np.random.rand() < math.exp(-delta / (T + 1e-12))
        if accept:
            state = candidate
            current_e = ce
        if ce < best_e:
            best = candidate.copy()
            best_e = ce
    return best, best_e

