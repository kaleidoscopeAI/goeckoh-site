for (let k = 0; k < m; k++) {
const v = new Float32Array(n);
const Dk = this.cfg.diffusion[k];
for (let i = 0; i < n; i++) v[i] = this.S[i * m + k] * Dk;
const Lv = sparseMul(this.graph.L!, v);
for (let i = 0; i < n; i++) out[i * m + k] += -Lv[i];
