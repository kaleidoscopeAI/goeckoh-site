for (let i = 0; i < this.graph.n; i++) {
const node = this.nodeStates[i];
const base = i * this.cfg.speciesCount;
for (let k = 0; k < this.cfg.speciesCount; k++) node.species[k] = this.S[base + k];
