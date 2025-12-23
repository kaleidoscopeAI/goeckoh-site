  for (let i = this.edges.length - 1; i >= 0; i--) {
    const edge = this.edges[i];
    if (edge.w < this.wMin) {
      this.removeEdge(edge.a, edge.b);
      continue;
    }
    const newW = Math.random();
    const deltaH = this.computeDeltaHForEdge(edge, newW);
    const p = Math.min(1, Math.exp(-deltaH / this.temperature));
    if (Math.random() < p) edge.w = newW;
  }
