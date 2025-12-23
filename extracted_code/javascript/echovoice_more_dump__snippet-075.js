function applyDelta(id: string, delta: number[]) {
  const node = engine.nodes.get(id);
  if (!node) return;
  for (let d = 0; d < Math.min(node.sem.length, delta.length); d++) {
    const clipped = Math.max(-0.5, Math.min(0.5, delta[d]));
    node.sem[d] += engine.llm_eta * clipped;
  }
