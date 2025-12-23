# test_core_er_epr.py
import numpy as np
from core_er_epr import RelationalCore, entropy_vn, build_simple_hamiltonians, build_jump_ops_from_stress

def test_rho_and_entropy():
    # simple Bell-like 2x2
    R = np.zeros((2,2), dtype=complex)
    R[0,0] = 1.0/np.sqrt(2)
    R[1,1] = 1.0/np.sqrt(2)
    core = RelationalCore(R)
    rhoS = core.rho_S()
    assert rhoS.shape == (2,2)
    S = core.entanglement_entropy()
    # for Bell pair S = ln(2)
    assert abs(S - np.log(2)) < 1e-8

def test_probs_and_correction():
    R = np.array([[0.8, 0.1],[0.2,0.9]], dtype=complex)
    core = RelationalCore(R)
    p_born = core.probs_born()
    p_prod = core.probs_softlogprod(alpha=1.0,beta=3.0)
    assert abs(p_born.sum() - 1.0) < 1e-12
    assert abs(p_prod.sum() - 1.0) < 1e-12
    pc = core.probability_with_er_correction(lam=0.5, mix_mode='born')
    assert abs(pc.sum() - 1.0) < 1e-12

def test_lindblad_transition():
    R = (np.random.randn(3,3)+1j*np.random.randn(3,3))
    core = RelationalCore(R)
    H = None
    Ls = build_jump_ops_from_stress(core, max_ops=3)
    newcore = core.lindblad_step_joint(H, Ls, dt=1e-3)
    assert newcore.R.shape == core.R.shape
