"""
Implements the mathematical models from the specifications
"""
@staticmethod
def node_growth_rate(N0: float, r: float, t: float) -> float:
    """N(t) = N0 * e^(rt): Node growth model"""
    return N0 * math.exp(r * t)

@staticmethod
def knowledge_growth(K0: float, k: float, c: float, t: float) -> float:
    """K(t) = K0 * e^((k+c)t): Knowledge accumulation"""
    return K0 * math.exp((k + c) * t)

@staticmethod
def system_failure_probability(p: float, n: int) -> float:
    """P(system failure) = p^n: Resilience calculation"""
    return p ** n

@staticmethod
def parallel_processing_time(Ts: float, P: float, N: int) -> float:
    """T_p = T_s * ((1 - P) + (P / N)): Amdahl's Law"""
    return Ts * ((1 - P) + (P / N))

@staticmethod
def learning_efficiency(E0: float, R: float, t: float) -> float:
    """E(t) = E0 * ln(1 + Rt): Learning convergence"""
    return E0 * math.log(1 + R * t)

