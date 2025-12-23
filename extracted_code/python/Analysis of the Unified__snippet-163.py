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

