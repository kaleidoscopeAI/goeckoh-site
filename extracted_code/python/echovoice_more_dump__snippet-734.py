def hajek_schedule(c: float = 1.0) -> Callable[[int], float]:
    def schedule(t: int) -> float:
        return float(c) / max(1e-12, math.log(max(2.0, float(t) + 1.0)))
    return schedule

def constant_schedule(T: float = 1.0) -> Callable[[int], float]:
    return lambda t: float(T)

