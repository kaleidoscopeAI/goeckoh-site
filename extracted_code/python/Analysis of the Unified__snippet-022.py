# lindblad_relax.py
"""
Lindblad-style dissipative embedding for stress equalization.
This module constructs Lindblad jump operators L_k from local stress patterns
to produce a master equation:
  dρ/dt = -i[H, ρ] + Σ_k (L_k ρ L_k^† - 1/2 {L_k^† L_k, ρ})

We provide utilities to:
 - convert R->ρ_joint = vec(R) vec(R)† (pure joint state)
 - build jump operators coupling basis elements with amplitude proportional to stress
 - integrate a short Lindblad step (Euler) for demonstration
Note: computationally expensive for dS*dA large (n^2 density matrix).
"""

from __future__ import annotations
import numpy as np
from typing import List, Tuple

def vec_to_rho_joint(R: np.ndarray) -> np.ndarray:
    """
    R shape (dS, dA) -> psi = vec(R) shape (n,1) with n=dS*dA
    -> rho = psi psi^†
    """
    psi = R.reshape(-1,1)
    rho = psi @ psi.conj().T
    return rho

def build_local_jump_ops(R: np.ndarray, Q: np.ndarray, B: np.ndarray, max_ops: int = 100) -> List[np.ndarray]:
    """
    Build a small list of jump operators L_k (n x n) that act on the joint Hilbert space.
    For each index (i,j), stress = |R_ij|^2 - |Q_ij|^2. If stress > 0, create an operator that
    moves amplitude away from that local basis vector |ij> to a sink or neighbor basis.
    This is heuristic: we build rank-1 operators |sink><ij|.
    """
    dS, dA = R.shape
    n = dS * dA
    I_R = np.abs(R)**2
    I_Q = np.abs(Q)**2
    stress = I_R - I_Q
    ops = []
    # create sink basis vector as uniform superposition of basis vectors (or one chosen)
    sink = np.ones((n,), dtype=complex) / np.sqrt(n)
    sink = sink.reshape((n,1))
    # flatten index mapping
    idxs = [(i,j) for i in range(dS) for j in range(dA)]
    # sort by absolute stress desc
    flat_stress = [(abs(stress[i,j]), i, j) for i,j in idxs]
    flat_stress.sort(reverse=True)
    for val,i,j in flat_stress[:max_ops]:
        if val <= 1e-8:
            break
        # create operator |sink><basis_ij|
        basis = np.zeros((n,1), dtype=complex)
        flat_idx = i * dA + j
        basis[flat_idx,0] = 1.0
        L = sink @ basis.conj().T  # n x n
        # scale operator by sqrt(B_ij * |stress|)
        scale = np.sqrt(max(0.0, B[i,j] * val))
        ops.append(scale * L)
    return ops

def lindblad_step(rho: np.ndarray, H: np.ndarray, Ls: List[np.ndarray], dt: float) -> np.ndarray:
    """
    Euler step for Lindblad master equation.
    """
    # unitary
    drho = -1j * (H @ rho - rho @ H)
    # dissipators
    for L in Ls:
        LrhoL = L @ rho @ L.conj().T
        LdagL = L.conj().T @ L
        drho += LrhoL - 0.5 * (LdagL @ rho + rho @ LdagL)
    rho_new = rho + dt * drho
    # ensure hermiticity and positivity small fixes
    rho_new = 0.5 * (rho_new + rho_new.conj().T)
    # renormalize trace
    tr = np.trace(rho_new)
    if tr == 0:
        # fallback small identity
        n = rho_new.shape[0]
        rho_new = np.eye(n, dtype=complex) / float(n)
    else:
        rho_new /= tr
    return rho_new
