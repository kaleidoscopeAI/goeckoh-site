const phi = this.computeIIT();
return {
  nodes: Array.from(this.nodes.values()).map(n => ({
    id: n.id, pos: n.pos, e_head: n.e.slice(0, 4), energy: n.energy, k: n.k, a: n.a, b: n.b, h: n.h,
    kappa: n.kappa, mu: n.mu, c: n.c, repProb: n.repProb, pruneRisk: n.pruneRisk
  })),
  edges: this.edges.map(e => ({ a: e.a, b: e.b, k: e.k, l: e.l, w: e.w })),
  stats: { totalH: this.computeH(), phi, totalEnergy: Array.from(this.nodes.values()).reduce((sum, n) => sum + n.energy, 0) }
};
