let faissIndex: Map<string, number[][]> = new Map(); // Simple array search
export function persistCrystal(engine: Engine, n: NodeState) {
  if (n.histVar < engine.crystalEpsX) {
    faissIndex.set(n.id, n.tensor); // Store subspace
  }
