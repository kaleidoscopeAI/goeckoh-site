if (this.nodes.size >= this.max_nodes) throw new Error("Max nodes reached");
const id = randomUUID();
const p: Vec3 = pos ?? [(Math.random()-0.5)*6, (Math.random()-0.5)*1, (Math.random()-0.5)*6];
const s = sem ?? new Array(this.D).fill(0).map(gaussian);
const n: NodeState = {
  id, pos: p, vel: [0,0,0], mass: 1, energy: 0, K: Math.random()*0.5+0.1, A: 0.2,
  sem: s, mutation_sigma: this.mutation_sigma_default, repProb: 0, neighbors: []
};
this.nodes.set(id, n);
return n;
