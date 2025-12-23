if (this.nodes.size >= this.maxNodes) throw new Error("Max nodes");
const id = randomUUID();
const p: Vec3 = pos ?? [(Math.random() - 0.5) * 6, (Math.random() - 0.5) * 1, (Math.random() - 0.5) * 6];
const bits = e ?? new Array(this.dBit).fill(0).map(() => Math.random() > 0.5 ? 1 : 0);
const n: NodeState = {
  id, pos: p, vel: [0, 0, 0], mass: 1, e: bits, energy: 0, k: 0.5, a: 0.2,
  b: gaussian() * 0.1, h: gaussian() * 0.1, kappa: gaussian() * 0.1, mu: gaussian() * 0.1,
  c: new Array(this.emotionalSpecies).fill(0.5), s: Math.random() > 0.5 ? 1 : -1,
  mutation_sigma: this.mutationSigmaDefault, repProb: 0, pruneRisk: 0, neighbors: []
};
this.nodes.set(id, n);
this.history.push([]);
return n;
