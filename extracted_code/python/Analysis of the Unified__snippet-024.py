"""
core_er_epr.py
Core Relational Quantum + ER=EPR engine (Part 1)

Provides:
- RelationalCore: holds R, Q (optional mirror), compute rho_S/rho_A, norms
- entropy_vn: von Neumann entropy (uses qutip if available)
- bridge_strength_map: compute per-element bridge strengths
- entanglement_corrected_R: apply ER correction to R
- hybrid_probs: compute mixed Born/product probabilities
- lindblad_step_joint: tiny Euler step for joint density w/ small set of L_k

Dependencies: numpy, scipy (optional), qutip (optional).
"""
from __future__ import annotations
import numpy as np
from typing import Tuple, Dict, Any, Optional, List
import math
import logging

logger = logging.getLogger("core_er_epr")
logging.basicConfig(level=logging.INFO)

# try optional imports
_try_qutip = True
try:
    import qutip as qt
except Exception:
    _try_qutip = False

try:
    import scipy.linalg as sla
except Exception:
    sla = None

# -----------------------------
# Numerical helpers
# -----------------------------
_eps = 1e-12

def safe_normalize_matrix(M: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(M)
    if norm <= 0:
        return M
    return M / norm

def eigvals_hermitian(mat: np.ndarray) -> np.ndarray:
    """Return eigenvalues for Hermitian matrix (real) robustly."""
    if sla is not None:
        # use eigh from scipy for stability if available
        try:
            vals = sla.eigvalsh(mat)
            return np.real(vals)
        except Exception:
            pass
    # fallback numpy
    vals, _ = np.linalg.eigh(mat)
    return np.real(vals)

def log_safe(x: np.ndarray) -> np.ndarray:
    return np.log(np.clip(x, _eps, None))

# -----------------------------
# Entropy utilities
# -----------------------------
def entropy_vn_from_rho(rho: np.ndarray) -> float:
    """
    Von Neumann entropy S(rho) = -Tr(rho log rho)
    Accepts numpy ndarray (Hermitian, trace=1)
    """
    # small Hermitian symmetrize
    rho_h = 0.5 * (rho + rho.conj().T)
    vals = eigvals_hermitian(rho_h)
    # clamp negative tiny eigenvalues
    vals = np.clip(vals, 0.0, None)
    if vals.sum() <= 0:
        return 0.0
    probs = vals / np.sum(vals)
    s = -np.sum(probs * log_safe(probs))
    return float(np.real(s))

# If qutip is available provide wrapper that uses qutip.entropy_vn for speed
def entropy_vn(rho: np.ndarray) -> float:
    if _try_qutip:
        try:
            qrho = qt.Qobj(rho)
            return float(qt.entropy_vn(qrho))
        except Exception:
            return entropy_vn_from_rho(rho)
    else:
        return entropy_vn_from_rho(rho)

# -----------------------------
# Relational core
# -----------------------------
class RelationalCore:
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

# -----------------------------
# convenience functions
# -----------------------------
def build_simple_hamiltonians(dS: int, dA: int, energy_scale: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build small Hs (dSxdS), Ha (dAxdA), and joint Hint (dS*dA x dS*dA) optionally.
    This is a utility used in examples; not a physics-unique choice.
    """
    def rand_herm(n):
        X = (np.random.randn(n,n) + 1j*np.random.randn(n,n)) / np.sqrt(2.0)
        H = X + X.conj().T
        H = H / np.max(np.abs(H)) * energy_scale
        return H
    Hs = rand_herm(dS) if dS>2 else np.array([[0.0,1.0],[1.0,0.0]], dtype=np.complex128)*energy_scale
    Ha = rand_herm(dA) if dA>2 else np.array([[0.5,0.0],[0.0,-0.5]], dtype=np.complex128)*(energy_scale*0.5)
    # simple Hint as Kronecker sum small coupling
    Hint = np.kron(Hs, Ha) * (0.01 * energy_scale)
    return Hs, Ha, Hint

# -----------------------------
# Example: build jump operators from stress
# -----------------------------
def build_jump_ops_from_stress(core: RelationalCore, max_ops: int = 20) -> List[np.ndarray]:
    """
    Construct rank-1 jump operators on joint space based on stress = |R|^2 - |Q|^2.
    Each operator moves amplitude from high-stress basis vector to a uniform sink.
    """
    dS, dA = core.R.shape
    n = dS * dA
    I_R = np.abs(core.R)**2
    I_Q = np.abs(core.Q)**2
    stress = I_R - I_Q
    flat = []
    for i in range(dS):
        for j in range(dA):
            flat.append((abs(stress[i,j]), i, j))
    flat.sort(reverse=True, key=lambda x: x[0])
    # sink vector uniform
    sink = np.ones((n,1), dtype=np.complex128) / np.sqrt(n)
    ops = []
    count = 0
    for val,i,j in flat:
        if count >= max_ops:
            break
        if val <= 1e-8:
            break
        idx = i*dA + j
        basis = np.zeros((n,1), dtype=np.complex128)
        basis[idx,0] = 1.0
        L = sink @ basis.conj().T  # rank-1
        scale = np.sqrt(max(0.0, val))
        ops.append(scale * L)
        count += 1
    return ops

# -----------------------------
# Demonstration / quick-run
# -----------------------------
def demo_small_run(dS=3, dA=3):
    # random R
    R = (np.random.randn(dS,dA) + 1j*np.random.randn(dS,dA)) * 0.5
    core = RelationalCore(R)
    print("dims:", core.dims)
    print("entanglement entropy S =", core.entanglement_entropy())
    print("bridge strengths:\n", core.bridge_strength_map(gamma=1.0))
    print("born probs:", core.probs_born())
    print("softprod probs:", core.probs_softlogprod(alpha=1.0,beta=2.0))
    print("hybrid probs:", core.measure_probs(mode='hybrid', mix=0.4, product_params={'alpha':1.0,'beta':2.0}))
    # apply entanglement correction
    Rcorr = core.entanglement_corrected_R(kappa=0.3)
    core2 = RelationalCore(Rcorr)
    print("S after correction:", core2.entanglement_entropy())
    # lindblad step example
    d = dS*dA
    H = np.eye(d, dtype=np.complex128) * 0.01
    Ls = build_jump_ops_from_stress(core, max_ops=6)
    core_after = core.lindblad_step_joint(H, Ls, dt=0.01)
    print("entropy after small lindblad step:", core_after.entanglement_entropy())
    return core, core2, core_after

if __name__ == "__main__":
    demo_small_run(3,3)
