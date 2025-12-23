const m = this.cfg.speciesCount;
const base = nodeIdx * m;
for (let k = 0; k < m; k++) this.externalInputs[base + k] = injections[k];
