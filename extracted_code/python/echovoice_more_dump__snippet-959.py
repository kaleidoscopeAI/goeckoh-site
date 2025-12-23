L0 Bit/Hardware: AND a ∧ b (bits: combine), OR a ∨ b (bits: union), XOR a ⊕ b (bits: diff), NOT ¬a (bits: invert), Sum = A ⊕ B ⊕ Cin (bits: add), Carry = (A ∧ B) ∨ (Cin ∧ (A ⊕ B)) (bits: propagate).
L1 Numerical: x = (-1)^s * (1 + m/2^52) * 2^(e-1023) (bits: float rep, pack/unpack).
L2 Vector Axioms: (x + y) + z = x + (y + z) (bits: vector add), x + y = y + x, x + 0 = x, x + (-x) = 0, α(x + y) = αx + αy (bits: scale via mul), etc. (bits: aggregate to vec).
L3 Algebra: A * B (bits: mat mul), det(A) (bits: Laplace exp), A^{-1} (bits: Gauss elim).
L4 Calculus: df/dx ≈ (f(x+ε)-f(x))/ε (bits: finite diff), ∫ f dx ≈ sum f_i Δx (bits: Riemann).
L5 Prob: P(A|B) = P(A∩B)/P(B) (bits: count ratios), E[X] = sum x_i p_i (bits: weighted sum).
L6 Reductions: Closed forms like sum 1/n = H_n ≈ ln n + γ (bits: harmonic approx).
L7 Dynamics: dx/dt = f(x) (bits: Euler step x += f*dt).
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

