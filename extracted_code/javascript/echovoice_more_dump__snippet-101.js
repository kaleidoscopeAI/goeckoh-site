  for (const n of this.nodes.values()) {
    const psiNorm = Math.sqrt(n.e.reduce((sum, v) => sum + v**2, 0));
    // Feedback to grads or tether
    this.alphaTether += 0.01 * psiNorm;
  }
