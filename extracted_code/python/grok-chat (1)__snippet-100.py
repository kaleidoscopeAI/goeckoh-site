def bond_energy(node1, node2):
    hamming = sum(b1 != b2 for b1, b2 in zip(node1.bits, node2.bits)) / 128
    dist = sum((p1 - p2)**2 for p1, p2 in zip(node1.position, node2.position))
    return 0.5 * hamming + 0.5 * dist

