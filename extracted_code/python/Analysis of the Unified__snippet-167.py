"""
Holds relational amplitude matrix R (dS x dA) and optional mirror Q.
Methods:
  - rho_S(), rho_A()
  - normalize_global()
  - measure_probs(mode='born'|'hybrid', **params)
  - entanglement_entropy()
  - bridge_map(...) -> per-element bridge strengths
  - entanglement_corrected_R(...)
"""
def __init__(self, R: np.ndarray, Q: Optional[np.ndarray] = None):
    R = np.asarray(R, dtype=np.complex128)
    if R.ndim != 2:
        raise ValueError("R must be 2D (dS x dA)")
    self.R = R.copy()
    dS, dA = self.R.shape
    if Q is None:
        self.Q = self.R.copy()  # default mirror identical (can be changed)
    else:
        self.Q = np.asarray(Q, dtype=np.complex128).reshape((dS, dA))
    # ensure nonzero normalization
    self.normalize_global()

@property
def dims(self) -> Tuple[int, int]:
    return self.R.shape

def normalize_global(self):
    """Normalize so joint vec(R) has unit norm (pure state)."""
    v = self.R.reshape(-1)
    norm = np.linalg.norm(v)
    if norm <= 0:
        # tiny random noise to avoid exact zero
        self.R += 1e-12 + 1e-12j
        norm = np.linalg.norm(self.R.reshape(-1))
        if norm <= 0:
            raise RuntimeError("Cannot normalize R")
    self.R /= norm
    # mirror normalization optional
    vq = self.Q.reshape(-1)
    nq = np.linalg.norm(vq)
    if nq <= 0:
        # set Q to be same direction as R
        self.Q = self.R.copy()
    else:
        self.Q /= nq

def rho_S(self) -> np.ndarray:
    """Return reduced density rho_S = R R^\dagger (dSxdS)"""
    return self.R @ self.R.conj().T

def rho_A(self) -> np.ndarray:
    """Return reduced density rho_A = R^\dagger R (dAxdA)"""
    return self.R.conj().T @ self.R

def entanglement_entropy(self) -> float:
    """Entropy of subsystem S (== subsystem A for pure joint state)."""
    rho_s = self.rho_S()
    return entropy_vn(rho_s)

# ------------------------
# Probabilities
# ------------------------
def probs_born(self) -> np.ndarray:
    I = np.abs(self.R) ** 2
    p = np.sum(I, axis=1)
    s = p.sum()
    if s <= 0:
        return np.ones_like(p) / float(p.size)
    return p / s

def probs_softlogprod(self, alpha: float = 1.0, beta: float = 1.0, eps: float = 1e-30) -> np.ndarray:
    I = np.abs(self.R) ** 2
    logs = np.sum(np.log(np.clip(I, eps, None)), axis=1) * float(alpha)
    logs = logs - np.max(logs)
    ex = np.exp(beta * logs)
    s = ex.sum()
    if s <= 0:
        return np.ones_like(ex) / float(ex.size)
    return ex / s

def measure_probs(self, mode: str = "born", mix: float = 0.0, product_params: Optional[Dict[str,float]] = None) -> np.ndarray:
    """
    mode: 'born' pure born; 'product' pure product via softlogprod; 'hybrid' mix via mix param
    mix in [0,1] where 0 = born, 1 = product
    product_params: pass alpha,beta to softlogprod
    """
    mix = float(np.clip(mix, 0.0, 1.0))
    if mode == "born":
        return self.probs_born()
    if mode == "product":
        pp = product_params or {'alpha':1.0,'beta':1.0}
        return self.probs_softlogprod(alpha=pp.get('alpha',1.0), beta=pp.get('beta',1.0))
    # hybrid
    p_born = self.probs_born()
    pp = product_params or {'alpha':1.0,'beta':1.0}
    p_prod = self.probs_softlogprod(alpha=pp.get('alpha',1.0), beta=pp.get('beta',1.0))
    p = (1.0 - mix) * p_born + mix * p_prod
    s = p.sum()
    if s <= 0:
        return np.ones_like(p) / float(p.size)
    return p / s

# ------------------------
# Bridge mapping (ER=EPR)
# ------------------------
def bridge_strength_map(self, gamma: float = 1.0) -> np.ndarray:
    """
    Compute per-element bridge scalar b_ij in [0,1].
    b_ij = |R_ij|^gamma * (S / Smax) normalized to [0,1]
    """
    dS, dA = self.R.shape
    S = self.entanglement_entropy()
    Smax = math.log(min(dS, dA) + _eps)
    # base amplitudes
    base = np.abs(self.R) ** float(gamma)
    # scale by entropy fraction
    factor = S / (Smax + _eps)
    raw = base * factor
    # normalize to [0,1]
    maxr = np.max(raw) if raw.size > 0 else 0.0
    if maxr <= 0:
        return np.zeros_like(raw)
    return raw / maxr

def entanglement_corrected_R(self, kappa: float = 0.2, offdiag_only: bool = True) -> np.ndarray:
    """
    Return R corrected by ER factor: R_ij -> R_ij*(1 + kappa*S/Smax)
    If offdiag_only True, apply only when i != j for system index (interpreting square R).
    """
    dS, dA = self.R.shape
    S = self.entanglement_entropy()
    Smax = math.log(min(dS, dA) + _eps)
    frac = S / (Smax + _eps)
    factor = 1.0 + float(kappa) * frac
    R2 = self.R.copy()
    if offdiag_only and dS == dA:
        for i in range(dS):
            for j in range(dA):
                if i != j:
                    R2[i,j] = R2[i,j] * factor
    else:
        R2 *= factor
    # re-normalize joint
    R2 = safe_normalize_matrix(R2)
    return R2

def probability_with_er_correction(self, lam: float = 1.0, mix_mode: str = "born", mix: float = 0.0, product_params: Optional[Dict[str,float]] = None) -> np.ndarray:
    """
    Compute base probabilities (born/hybrid) then add entanglement correction:
    p -> p + lam * (S/Smax) * (row_energy / total_energy)
    where row_energy = sum_j |R_ij|^2.
    """
    base = self.measure_probs(mode=mix_mode, mix=mix, product_params=product_params)
    I = np.abs(self.R) ** 2
    row_energy = np.sum(I, axis=1)
    total = np.sum(row_energy)
    if total <= 0:
        return base
    S = self.entanglement_entropy()
    Smax = math.log(min(self.R.shape) + _eps) if isinstance(self.R.shape, int) else math.log(min(self.R.shape) + _eps)
    # avoid degenerate; use min(dimS,dimA)
    dS, dA = self.R.shape
    Smax = math.log(min(dS,dA) + _eps)
    corr = (S / (Smax + _eps)) * (row_energy / total)
    p = base + float(lam) * corr
    s = p.sum()
    if s <= 0:
        return np.ones_like(p) / float(p.size)
    return p / s

# ------------------------
# Joint density and Lindblad
# ------------------------
def joint_state_vec(self) -> np.ndarray:
    """Return vec(R) as column vector (n,1) with n = dS*dA"""
    return self.R.reshape(-1, order='C').reshape(-1,1)

def joint_density(self) -> np.ndarray:
    psi = self.joint_state_vec()
    rho = psi @ psi.conj().T
    return rho

def lindblad_step_joint(self, H: Optional[np.ndarray], Ls: List[np.ndarray], dt: float = 1e-3, hbar: float = 1.0) -> np.ndarray:
    """
    Euler step for joint density with Lindblad dissipators.
    H is joint Hamiltonian (n x n) matching vec space, else zeros.
    Ls: list of jump operators (n x n) acting on joint space.
    Returns new joint density matrix.
    """
    rho = self.joint_density()
    n = rho.shape[0]
    if H is None:
        H = np.zeros((n,n), dtype=np.complex128)
    H = 0.5*(H + H.conj().T)
    # unitary piece
    drho = -1j / hbar * (H @ rho - rho @ H)
    # dissipators
    for L in Ls:
        LrhoL = L @ rho @ L.conj().T
        LdagL = L.conj().T @ L
        drho += LrhoL - 0.5 * (LdagL @ rho + rho @ LdagL)
    rho_new = rho + dt * drho
    rho_new = 0.5*(rho_new + rho_new.conj().T)
    tr = np.trace(rho_new)
    if abs(tr) < _eps:
        # fallback to maximally mixed
        rho_new = np.eye(n, dtype=np.complex128) / float(n)
    else:
        rho_new = rho_new / tr
    # attempt to recover R from (approx) pure rho if near pure: take dominant eigenvector
    vals, vecs = np.linalg.eigh(rho_new)
    idx = np.argmax(vals)
    psi = vecs[:, idx].reshape(-1,1)
    # reshape back into R shape
    dS, dA = self.R.shape
    newR = psi.reshape((dS, dA), order='C')
    core = RelationalCore(newR, Q=self.Q.copy())
    core.normalize_global()
    return core

