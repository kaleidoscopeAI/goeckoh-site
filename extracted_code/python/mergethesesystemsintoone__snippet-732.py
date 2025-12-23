@staticmethod
def redistribute_energy(nodes: List[PythonNode]):
    """Redistribute energy using a cubic spline."""
    energies = [node.energy for node in nodes]
    spline = CubicSpline(range(len(energies)), energies)
    redistributed = spline(np.linspace(0, len(energies) - 1, len(energies)))

    # Normalize the redistributed energy to avoid spikes or dips
    min_energy = min(redistributed)
    max_energy = max(redistributed)
    if max_energy - min_energy > 0:
        redistributed = (redistributed - min_energy) / (max_energy - min_energy) * (15 - 5) + 5

    for i, node in enumerate(nodes):
        node.energy = max(redistributed[i], 0)
    logging.info("Energy redistributed among nodes.")

