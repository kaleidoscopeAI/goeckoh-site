from core_er_epr import RelationalCore
import numpy as np
R = np.array([[0.8,0.1],[0.2,0.9]], dtype=complex)
c = RelationalCore(R)
print("Born:", c.probs_born())
print("SoftProduct:", c.probs_softlogprod(alpha=1.0,beta=4.0))
print("Hybrid (mix=0.5):", c.measure_probs(mode='hybrid', mix=0.5, product_params={'alpha':1.0,'beta':4.0}))
print("Entanglement S:", c.entanglement_entropy())
print("Bridge map:", c.bridge_strength_map())
