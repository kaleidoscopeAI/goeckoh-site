class Spec:
    SelFn = Callable[["HybridState", int], Sequence[int]]

def hajek_schedule(c: float = 1.0) -> Callable[[int], float]:
    """Annealing schedule: temperature decreases slowly over time."""
    def schedule(t: int) -> float:
        return float(c) / max(1e-12, math.log(max(2.0, float(t) + 1.0)))
    return schedule

