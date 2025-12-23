def __init__(self, core: RelationalCore):
    self.core = core

def compute(self) -> Dict[str, float]:
    rho = self.core.rho_S()
    purity = float(np.real(np.trace(rho @ rho)))
    entropy = self.core.entanglement_entropy()
    max_stress = float(np.max(np.abs(np.abs(self.core.R)**2 - np.abs(self.core.Q)**2)))
    avg_bridge = float(np.mean(self.core.bridge_strength_map(gamma=1.0)))
    probs = self.core.probs_born()
    # signal/noise as ratio of top row energy vs rest
    row_energy = np.sum(np.abs(self.core.R)**2, axis=1)
    snr = float((np.max(row_energy) + 1e-12) / (np.mean(row_energy) + 1e-12))
    return {'purity': purity, 'entropy': entropy, 'max_stress': max_stress, 'avg_bridge': avg_bridge, 'snr': snr, 'probs_sum': float(probs.sum())}

