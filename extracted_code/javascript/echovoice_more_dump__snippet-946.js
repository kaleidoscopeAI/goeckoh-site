for (const n of [...this.nodes.values()]) {
  if (Math.random() < n.pruneRisk) {
    this.removeNode(n.id);
  }
}
this.edges = this.edges.filter(e => this.nodes.has(e.a) && this.nodes.has(e.b));
