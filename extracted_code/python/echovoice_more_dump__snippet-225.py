All equations from docs, with bit changes: Low levels mutate bits (e.g., L0 AND/OR combine for semantic ops), propagate up (L7 dynamics evolve pos from bit-derived E).

    Node: x_i ∈ ℝ^3, s_i (embedding vec), E_i (tension), K_i (conf), P_i (bias vec), A_i (attention).
    Evolution: ṡ_i = η Σ w_ij (s_j - s_i) + u_LLM + ξ (diffusion/mutation/LLM; bits: gaussian ξ flips probs).
    H = Σ [λ_bit(1-sim_bit) + λ_pos||x_i-x_j||^2 + w_ij attr] + Σ α|x_i-x_{i0}|^2 + H_reg (global min; bits: sim_bit =1-popcount(E_i⊕E_j)/128).
    FRF ∂g_ij/∂t = -2 R_ij + κ Teff_ij (R_ij = d_i+d_j-2; bits: smooth adj mat A_ij).
    SDE dx = f dt + g dW (bits: Wiener dW gaussian on pos bits).
    Torque τ = novelty * ar (novelty=1-cos sim; gates if >0.5).
    V-A-S: v̇ = gaussian * ar, ar = |E|, st = K-0.5, coh=1-|v-st| (bits: emotional feedback to bit flips).
    E = f(E,coh,v) - λ A cos(Δϕ) (physics layer; bits: coherence from bit entropy).
    Levels 0-19: e.g., L0 a⊕b (bit diff for mutation), L8 P=|<\i|ψ>|^2 (prob from amps for sim_bit), full flow as before.
    Other: dρ/dt = -i[H,ρ] + Lindblad (decoh; bits: matrix on quantized pos).

