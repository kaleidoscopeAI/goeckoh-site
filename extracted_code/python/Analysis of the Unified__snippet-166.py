"""Return eigenvalues for Hermitian matrix (real) robustly."""
if sla is not None:
    # use eigh from scipy for stability if available
    try:
        vals = sla.eigvalsh(mat)
        return np.real(vals)
    except Exception:
        pass
# fallback numpy
vals, _ = np.linalg.eigh(mat)
return np.real(vals)

