"""
Measures "aliveness" via the life functional:

L_k = λ1 * (-ΔS_int / ΔS_env) + λ2 * I(X;E)/H(X) + λ3 * (ΔN/N)

Terms:
1. Thermodynamic self-maintenance
2. Predictive information
3. Reproductive/persistence drive
"""

def __init__(self, window_size: int = 20):
    self.window_size = window_size

    # Coefficients
    self.lambda1 = 1.0  # Thermodynamic weight
    self.lambda2 = 1.0  # Predictive weight
    self.lambda3 = 0.5  # Reproductive weight

    # History buffers
    self.entropy_int_history = deque(maxlen=window_size)
    self.entropy_env_history = deque(maxlen=window_size)
    self.viable_count_history = deque(maxlen=window_size)

    # Feature windows for prediction
    self.internal_features = deque(maxlen=window_size)
    self.external_features = deque(maxlen=window_size)

def compute_entropy(self, distribution: np.ndarray) -> float:
    """Shannon entropy of a distribution"""
    p = distribution + 1e-10  # Avoid log(0)
    p = p / p.sum()
    return float(-np.sum(p * np.log(p)))

def compute_life_intensity(
    self,
    internal_state: np.ndarray,
    external_state: np.ndarray,
    viable_nodes: int
) -> float:
    """
    Compute L_k for current step

    Args:
        internal_state: Current internal feature vector
        external_state: Current external/environment features
        viable_nodes: Count of "alive" nodes/patterns
    """
    # --- Term 1: Thermodynamic self-maintenance ---
    # Internal entropy (bit distribution)
    S_int = self.compute_entropy(internal_state)
    self.entropy_int_history.append(S_int)

    # Environmental entropy
    S_env = self.compute_entropy(external_state)
    self.entropy_env_history.append(S_env)

    # Compute derivatives
    if len(self.entropy_int_history) > 1:
        dS_int = S_int - self.entropy_int_history[-2]
        dS_env = S_env - self.entropy_env_history[-2]
        T1 = -dS_int / (dS_env + 1e-6)
    else:
        T1 = 0.0

    # --- Term 2: Predictive information ---
    self.internal_features.append(internal_state)
    self.external_features.append(external_state)

    if len(self.internal_features) >= 5:
        # Correlation between recent internal and future external
        int_window = np.array(list(self.internal_features)[-5:])
        ext_window = np.array(list(self.external_features)[-5:])

        correlation = np.corrcoef(
            int_window.flatten(),
            ext_window.flatten()
        )[0, 1]

        H_eff = max(S_int, 1e-6)
        T2 = correlation / H_eff
    else:
        T2 = 0.0

    # --- Term 3: Reproductive drive ---
    self.viable_count_history.append(viable_nodes)

    if len(self.viable_count_history) > 1:
        N_k = viable_nodes
        N_prev = self.viable_count_history[-2]
        dN = N_k - N_prev
        T3 = dN / max(N_k, 1)
    else:
        T3 = 0.0

    # --- Combine ---
    L_k = (
        self.lambda1 * T1 +
        self.lambda2 * T2 +
        self.lambda3 * T3
    )

    return float(L_k)


