for (const n of this.nodes.values()) {
  const k = Math.floor(Math.random() * this.dBit); // Random bit to flip
  const oldE = n.e[k];
  n.e[k] = this.bitProbabilistic ? clip(gaussian() * 0.1 + oldE, 0, 1) : 1 - oldE;
  const deltaH = this.computeDeltaHForBit(n.id, k); // Incremental
  const pAccept = Math.min(1, Math.exp(-deltaH / this.temperature));
  if (Math.random() > pAccept) n.e[k] = oldE; // Reject
}
