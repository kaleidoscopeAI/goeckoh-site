const newE = new Map<string, number[]>();
for (const n of this.nodes.values()) newE.set(n.id, n.e.slice());
for (const n of this.nodes.values()) {
  const accum = new Array(this.dBit).fill(0);
  const numNeigh = n.neighbors.length || 1;
  for (const neighId of n.neighbors) {
    const nb = this.nodes.get(neighId)!.e;
    for (let d = 0; d < this.dBit; d++) accum[d] += (nb[d] - n.e[d]) / numNeigh;
  }
  const out = newE.get(n.id)!;
  for (let d = 0; d < this.dBit; d++) {
    out[d] += this.etaSem * accum[d] + gaussian() * n.mutation_sigma;
    if (this.bitProbabilistic) out[d] = clip(out[d], 0, 1);
    else out[d] = Math.random() < out[d] ? 1 : 0; // Threshold if soft
  }
}
for (const [id, arr] of newE) this.nodes.get(id)!.e = arr;
