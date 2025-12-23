Hamiltonian H = Σ_{(i,j)} [λ_bit (1 - sim_bit(E_i,E_j)) + λ_pos |x_i - x_j|^2 + w_ij attr(i,j)] + Σ_i α_i |x_i - x_{i0}|^2 + H_reg (bits: sim_bit Hamming, changes via flips).
sim_bit = 1 - popcount(E_i ⊕ E_j)/128 or (1/128) Σ p_k^i p_k^j (2δ -1) (bits: XOR/count or prob mul).
Master dS/dt = -i [H_total, S] + D_emotional(S) + D_decoh(S) + η_noise (bits: commutator approx via mat ops).
Position m_i \ddot{x}i = -∇{x_i} H - γ_i \dot{x}_i + ξ_i (bits: grad on pos, noise gaussian).
Lindblad dρ/dt = -i [H_q, ρ] + Σ (L_k ρ L_k^† - 1/2 {L_k^† L_k, ρ}) (bits: dephasing L=√γ Z).
Engines db_i/dt = α_p I_i o_i - β_p b_i + γ_p Σ w_ij (b_j - b_i) + ζ (similar for h, κ, μ) (bits: diffuse states).
FRF ∂g_ij/∂t = -2 R_ij (bits: edge curvature R_ij = d_i + d_j - 2 A_ij, smooth).
SDE dx_i = f(x_i) dt + g(x_i) dW_i (bits: Wiener dW gaussian).
Teff_ij from ψ, ∂g_ij/∂t = -2 R_ij + κ Teff_ij (bits: tension feedback to edges).
Torque τ = novelty * arousal (bits: dist * intensity, gate).
V-A-S: valence ±1, arousal 0-1, stance ±1, coherence (bits: vector c update).
Complexity O(N) per step (bits: local ops ensure).

