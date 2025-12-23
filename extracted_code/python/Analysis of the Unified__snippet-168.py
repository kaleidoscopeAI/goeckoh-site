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

