for (const edge of [...this.edges]) {
  if (edge.w < this.wMin) {
    this.removeEdge(edge.a, edge.b);
  } else {
    // Propose toggle w, accept if ΔH <0 or Metropolis
    const oldW = edge.w;
    edge.w = Math.random();
    const deltaH = this.computeH() - this.computeH(); // Full, optimize
    const p = Math.min(1, Math.exp(-deltaH / this.temperature));
    if (Math.random() > p) edge.w = oldW;
  }
}
