const base = i * this.cfg.speciesCount;
for (let k = 0; k < this.cfg.speciesCount; k++) node.species[k] = this.S[base + k];
