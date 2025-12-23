const newS = new Map<string, number[]>();
for (const [id, n] of this.nodes) newS.set(id, n.sem.slice());
for (const [id, n] of this.nodes) {
  const neigh = n.neighbors;
  const accum = new Array(this.D).fill(0);
  for (const j of neigh) {
    const sj = this.nodes.get(j)!.sem;
    for (let d = 0; d < this.D; d++) accum[d] += (sj[d] - n.sem[d]) / Math.max(1, neigh.length);
  }
  const out = newS.get(id)!;
  for (let d = 0; d < this.D; d++) out[d] = n.sem[d] + this.eta_sem * accum[d] + gaussian() * n.mutation_sigma;
}
for (const [id, arr] of newS) this.nodes.get(id)!.sem = arr;
