    L8 Quantum: |ψ> = α|0> + β|1> (bits: complex vec), <ψ|ψ> = 1 (bits: norm), P = |<i|ψ>|^2 (bits: prob from amp).
    L9 Info: H = -sum p log p (bits: entropy), D(KL) = sum p log(p/q) (bits: divergence).
    L10 Topology: χ = V - E + F (bits: Euler char).
    L11 Geometry: d(x,y) = ||x-y|| (bits: dist), κ = 1/r (bits: curvature).
    L12 Group: g * h (bits: compose), e * g = g (bits: identity).
    L13 Category: f ∘ g (bits: func chain).
    L14 Logic: ∀x P(x) (bits: quant), ∃x P(x).
    L15 Set: A ∪ B, A ∩ B (bits: union/intersect).
    L16 Graph: A_ij (bits: adj mat), d(v) = sum A_vj (bits: degree).
    L17 Game: U_i(s) (bits: utility), NE where no deviate (bits: fixed point).
    L18 Control: x' = Ax + Bu (bits: state space).
    L19 Cognition: High-level like E = f(energy, valence) (bits: reduce to scalar).

Bit Changes Flow: Starts at L0 (raw ops mutate single bits), L1 (pack bits to nums), L2 (nums to vecs, ops propagate changes), L3 (vecs to mats, mul transforms clusters), L4-5 (diff/prob on mats, stochastic flips), L6 (reduce to scalars, stabilize), L7 (time evolve, continuous change), L8 (quantum amps from probs, interference), up to L19 (cognitive E from all, feedback to bit flips via annealing).

