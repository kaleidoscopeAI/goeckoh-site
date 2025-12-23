# relational_hybrid_engine.py
"""
Hybrid Relational Engine
- Blends Born-sum and Product (π) probability rules via a tunable mixing parameter.
- Adds mirrored matrix Q and stress-bond dynamics that try to equalize intensities.
- Provides a simple, safe AI-control loop (pluggable) that adjusts mix & bond strengths
  based on diagnostics (no external LLM required; a stub interface is provided).
- Includes a demo `run_demo()` that runs a small 2x2 example and writes snapshots.

Usage:
    python relational_hybrid_engine.py  # runs demo
Dependencies:
    numpy
Optional hooks:
    - integrate with your RelationalEngine by replacing RelationalCore <-> HybridRelationalCore.
"""

from __future__ import annotations
import numpy as np
import math
import time
import logging
from typing import Optional, Callable, Dict, Any, Tuple

logger = logging.getLogger("hybrid_relational")
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", "%H:%M:%S"))
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# ---- Utilities ----
def safe_log(x: np.ndarray, eps: float = 1e-30) -> np.ndarray:
    return np.log(np.clip(x, eps, None))

def normalize_vector(v: np.ndarray) -> np.ndarray:
    s = float(np.sum(v))
    if s == 0:
        return np.ones_like(v) / float(v.size)
    return v / s

# ---- Hybrid probability rules ----
def probs_born(R: np.ndarray) -> np.ndarray:
    """Standard Born rule marginalization: p_i ∝ Σ_j |R_ij|^2"""
    I = np.abs(R) ** 2
    p = np.sum(I, axis=1)
    return normalize_vector(p)

def probs_product(R: np.ndarray) -> np.ndarray:
    """Raw product rule: p_i ∝ Π_j |R_ij|^2 (dangerous if zeros)"""
    I = np.abs(R) ** 2
    prod = np.prod(I, axis=1)
    # avoid all zeros
    if np.all(prod == 0):
        # fallback to small epsilon to avoid singularities
        prod = np.prod(np.clip(I, 1e-12, None), axis=1)
    return normalize_vector(prod)

def probs_geometric(R: np.ndarray) -> np.ndarray:
    """Geometric mean of intensities (soft product)"""
    I = np.abs(R) ** 2
    prod = np.prod(I, axis=1)
    gm = prod ** (1.0 / float(I.shape[1]))
    return normalize_vector(gm)

def probs_softlogprod(R: np.ndarray, alpha: float = 1.0, beta: float = 1.0) -> np.ndarray:
    """
    Numerically stable soft-product:
    logscore_i = alpha * sum_j log(|R_ij|^2)
    p_i ∝ exp(beta * logscore_i)
    alpha controls power (0 -> uniform, 1 -> product), beta controls sharpness.
    """
    I = np.abs(R) ** 2
    logs = np.sum(safe_log(I), axis=1) * float(alpha)
    # stabilize:
    logs = logs - np.max(logs)
    ex = np.exp(beta * logs)
    return normalize_vector(ex)

# ---- Hybrid mix function ----
def mix_probs(R: np.ndarray, mix: float, method_product: str = "softlogprod",
              product_params: Optional[Dict[str, float]] = None) -> np.ndarray:
    """
    mix in [0,1]: 0 => pure Born (sum), 1 => pure product variant.
    method_product: 'product'|'geom'|'softlogprod'
    product_params: dict passed to probs_softlogprod (alpha,beta)
    """
    p_born = probs_born(R)
    if method_product == "product":
        p_prod = probs_product(R)
    elif method_product == "geom":
        p_prod = probs_geometric(R)
    else:
        pp = product_params or {}
        p_prod = probs_softlogprod(R, alpha=float(pp.get("alpha", 1.0)), beta=float(pp.get("beta", 1.0)))
    # convex mix
    mix = float(np.clip(mix, 0.0, 1.0))
    p = (1.0 - mix) * p_born + mix * p_prod
    return normalize_vector(p)

# ---- Hybrid Relational Core ----
class HybridRelationalCore:
    """
    Holds R (dS x dA) and mirrored Q (same shape).
    Manages bonds B (dS x dA) — nonnegative matrix of bond strengths connecting R_ij <-> Q_ij.
    mix ∈ [0,1] controls probability blend (0=Born,1=Product).
    gamma controls dissipative stress rate.
    """

    def __init__(self, R: np.ndarray, Q: Optional[np.ndarray] = None,
                 bonds: Optional[np.ndarray] = None,
                 mix: float = 0.0, gamma: float = 0.1,
                 product_method: str = "softlogprod", product_params: Optional[Dict[str,float]] = None):
        R = np.asarray(R, dtype=np.complex128)
        if R.ndim != 2:
            raise ValueError("R must be 2D array (dS,dA)")
        self.R = R.copy()
        dS, dA = self.R.shape
        if Q is None:
            # mirror default: copy R (symmetric start) with small perturbation
            self.Q = self.R.copy()
        else:
            self.Q = np.asarray(Q, dtype=np.complex128).reshape((dS, dA))
        if bonds is None:
            # initialize bonds proportional to average intensity of R columns
            B = np.abs(self.R)**2 + np.abs(self.Q)**2
            B = np.sum(B, axis=0)  # per-column base
            # broadcast to matrix
            B_mat = np.ones((dS, dA)) * (np.mean(B) * 0.5)
            self.B = B_mat
        else:
            self.B = np.asarray(bonds, dtype=float).reshape((dS, dA))
            self.B[self.B < 0] = 0.0
        self.mix = float(np.clip(mix, 0.0, 1.0))
        self.gamma = float(max(0.0, gamma))
        self.product_method = product_method
        self.product_params = product_params or {"alpha": 1.0, "beta": 1.0}
        # normalization
        self.normalize_global()

    # -----------------------
    # Basic helpers
    # -----------------------
    def normalize_global(self):
        norm = np.linalg.norm(self.R)
        if norm == 0:
            raise RuntimeError("Cannot normalize zero R")
        self.R /= norm
        # normalize Q as well
        normq = np.linalg.norm(self.Q)
        if normq == 0:
            self.Q = np.ones_like(self.Q, dtype=np.complex128) * (1.0 / math.sqrt(self.Q.size))
        else:
            self.Q /= normq

    def reduced_density_S(self) -> np.ndarray:
        return self.R @ self.R.conj().T

    def measure_probs(self) -> np.ndarray:
        """Return mixed probabilities according to current mix parameter."""
        return mix_probs(self.R, self.mix, method_product=self.product_method, product_params=self.product_params)

    def diagnostics(self) -> Dict[str, Any]:
        rho = self.reduced_density_S()
        purity = float(np.real(np.trace(rho @ rho)))
        entropy = -float(np.sum(np.clip(np.linalg.eigvalsh(rho), 1e-15, None) * np.log(np.clip(np.linalg.eigvalsh(rho), 1e-15, None))))
        avg_bond = float(np.mean(self.B))
        max_stress = float(np.max(np.abs(np.abs(self.R)**2 - np.abs(self.Q)**2)))
        return {"dims": self.R.shape, "mix": self.mix, "purity": purity, "entropy": entropy,
                "avg_bond": avg_bond, "max_stress": max_stress}

    # -----------------------
    # Stress gradient (dissipative)
    # -----------------------
    def stress_gradient(self) -> np.ndarray:
        """
        Compute G_R = Σ_{uv} B_{ij,uv} (|R_ij|^2 - |Q_uv|^2) R_ij
        Simplified to element-wise bonds B_{ij} connecting to same index in Q:
        G_R_ij = B_ij * (|R_ij|^2 - |Q_ij|^2) * R_ij
        """
        I_R = np.abs(self.R)**2
        I_Q = np.abs(self.Q)**2
        diff = I_R - I_Q
        return self.B * diff * self.R

    # -----------------------
    # Evolution step: unitary + stress (explicit integrator RK4 wrapper)
    # -----------------------
    def unitary_rhs(self, Hs: np.ndarray, Ha: np.ndarray, Hint: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Return -i/ħ * (Hs R - R Ha + Hint_R) flattened as matrix RHS.
        Here Hint if provided is expected to be precomputed operator that maps R->matrix (shape matches dS,dA).
        """
        # minimal validation
        if Hs is None or Ha is None:
            raise ValueError("Hs and Ha are required for unitary_rhs")
        # Hs @ R - R @ Ha
        return -1j * (Hs @ self.R - self.R @ Ha) if Hint is None else -1j * (Hs @ self.R - self.R @ Ha + Hint @ self.R)

    def evolve_step(self, Hs: np.ndarray, Ha: np.ndarray, Hint: Optional[np.ndarray], dt: float, rk4: bool = True):
        """
        Single integration step combining unitary and dissipative stress term.
        Uses operator splitting: first unitary (RK4 or expm approximation) then dissipative update.
        For simplicity we use RK4 for whole combined RHS (works well for small sizes).
        """
        dt = float(dt)
        ħ = 1.0
        # define combined RHS function for RK4
        def rhs(Rmat: np.ndarray, Qmat: np.ndarray):
            # unitary part
            U = -1j / ħ * (Hs @ Rmat - Rmat @ Ha)
            # hint term if provided: we interpret Hint as linear operator in big space; for simplicity allow Hint as matrix same shape
            if Hint is not None:
                U += -1j / ħ * (Hint @ Rmat)
            # dissipative stress part
            I_R = np.abs(Rmat)**2
            I_Q = np.abs(Qmat)**2
            diff = I_R - I_Q
            D = - self.gamma * (self.B * diff * Rmat)
            return U + D

        # RK4 steps
        R0 = self.R.copy()
        Q0 = self.Q.copy()
        k1 = rhs(R0, Q0)
        k2 = rhs(R0 + 0.5*dt*k1, Q0 + 0.5*dt*0)  # Q static in this step (we'll evolve Q separately)
        k3 = rhs(R0 + 0.5*dt*k2, Q0 + 0.5*dt*0)
        k4 = rhs(R0 + dt*k3, Q0 + dt*0)
        self.R = R0 + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)

        # Simple update rule for Q: let Q mirror R with slow dynamics (relaxation)
        # dQ/dt = -gamma_q * (I_Q - I_R) * Q  (opposite sign)
        gamma_q = self.gamma * 0.5
        I_R = np.abs(self.R)**2
        I_Q = np.abs(self.Q)**2
        self.Q = self.Q - dt * (gamma_q * (I_Q - I_R) * self.Q)

        # renormalize occasionally to prevent drift
        self.normalize_global()

    # -----------------------
    # Adaptive bond update rules
    # -----------------------
    def adapt_bonds_hebb(self, eta: float = 1e-3, decay: float = 1e-4):
        """
        Simple Hebbian-style bond adaptation:
        B_ij ← (1 - decay) * B_ij + eta * |I_R_ij - I_Q_ij| * B_ij
        This strengthens bonds that carry persistent stress.
        """
        I_R = np.abs(self.R)**2
        I_Q = np.abs(self.Q)**2
        stress = np.abs(I_R - I_Q)
        self.B = (1.0 - decay) * self.B + eta * stress * self.B
        # clamp
        self.B = np.clip(self.B, 0.0, None)

    # -----------------------
    # Simple AI-control hook
    # -----------------------
    def ai_control_step(self, controller: Callable[[Dict[str,Any]], Dict[str,Any]]):
        """
        controller: function that accepts diagnostics dict, returns adjustments:
            {
              "mix_delta": float (additive),
              "gamma_delta": float,
              "bond_scale": float (multiplicative),
              "product_params": {alpha, beta}
            }
        The function is safe-guarded: outputs are clamped.
        """
        try:
            diag = self.diagnostics()
            action = controller(diag)
            if not isinstance(action, dict):
                return
            mix_delta = float(action.get("mix_delta", 0.0))
            gamma_delta = float(action.get("gamma_delta", 0.0))
            bond_scale = float(action.get("bond_scale", 1.0))
            product_params = action.get("product_params", None)
            # apply with safety clamps
            self.mix = float(np.clip(self.mix + mix_delta, 0.0, 1.0))
            self.gamma = float(np.clip(self.gamma + gamma_delta, 0.0, 10.0))
            if not math.isfinite(bond_scale) or bond_scale <= 0.0:
                bond_scale = 1.0
            self.B = np.clip(self.B * bond_scale, 0.0, 1e6)
            if isinstance(product_params, dict):
                a = float(product_params.get("alpha", self.product_params.get("alpha",1.0)))
                b = float(product_params.get("beta", self.product_params.get("beta",1.0)))
                # clamp
                a = np.clip(a, 0.0, 5.0)
                b = np.clip(b, 0.0, 10.0)
                self.product_params["alpha"] = a
                self.product_params["beta"] = b
        except Exception as e:
            logger.warning("ai_control_step failure: %s", e)

# -----------------------
# Demo & test harness
# -----------------------
def demo_run(steps: int = 200, dt: float = 0.01, noisy: bool = True, log_every: int = 20):
    """
    Run a small 2x2 demo demonstrating hybrid mixing and stress dynamics.
    Saves diagnostics to a list and returns final core.
    """
    # initial R (two system outcomes, two apparatus columns)
    R0 = np.array([[0.8+0.0j, 0.1+0.0j],
                   [0.2+0.0j, 0.9+0.0j]], dtype=np.complex128)
    core = HybridRelationalCore(R0, mix=0.0, gamma=0.5, product_method="softlogprod", product_params={"alpha":1.0,"beta":3.0})
    # simple Hamiltonians (Pauli-like)
    Hs = np.array([[0.0, 1.0],[1.0, 0.0]], dtype=np.complex128)
    Ha = np.array([[0.5, 0.0],[0.0, -0.5]], dtype=np.complex128)
    Hint = None

    history = []
    # simple controller: gradually increase mix if max_stress > thresh; otherwise reduce mix slowly
    def simple_controller(diag):
        adv = 0.0
        if diag["max_stress"] > 0.1:
            adv = 0.02
        else:
            adv = -0.005
        # adjust gamma based on purity to stabilize
        gamma_delta = 0.0
        if diag["purity"] < 0.95:
            gamma_delta = 0.01
        return {"mix_delta": adv, "gamma_delta": gamma_delta, "bond_scale": 1.0, "product_params": {"alpha":1.0, "beta":3.0}}

    for t in range(steps):
        # optionally perturb R with small noise to simulate streaming data
        if noisy and (t % 15 == 0):
            noise = (np.random.randn(*core.R.shape) + 1j * np.random.randn(*core.R.shape)) * 0.005
            core.R += noise
        # evolve
        core.evolve_step(Hs, Ha, Hint, dt)
        # adapt bonds
        core.adapt_bonds_hebb(eta=1e-4, decay=1e-5)
        # ai step every few iterations
        if t % 5 == 0:
            core.ai_control_step(simple_controller)
        if t % log_every == 0 or t == steps-1:
            diag = core.diagnostics()
            p = core.measure_probs()
            history.append({"t": t*dt, "diag": diag, "p": p.copy(), "R": core.R.copy(), "Q": core.Q.copy(), "B": core.B.copy()})
            logger.info("t=%.3f mix=%.3f purity=%.4f entropy=%.4f probs=%s", t*dt, core.mix, diag["purity"], diag["entropy"], np.array2string(p, precision=3))
    return core, history

# Run demo if executed
if __name__ == "__main__":
    core, history = demo_run(steps=200, dt=0.01, noisy=True)
    # save a quick textual summary
    import json
    print("Final diagnostics:", history[-1]["diag"])
    with open("hybrid_demo_history.json", "w") as f:
        # serialize numeric arrays approximately
        out = [{"t":h["t"], "diag":h["diag"], "p":[float(x) for x in h["p"]]} for h in history]
        json.dump(out, f, indent=2)
    print("Wrote hybrid_demo_history.json")
