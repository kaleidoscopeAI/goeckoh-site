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

