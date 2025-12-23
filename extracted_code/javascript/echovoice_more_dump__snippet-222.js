const v = new Float32Array(n);
const Dk = this.cfg.diffusion[k];
for (let i = 0; i < n; i++) v[i] = this.S[i * m + k] * Dk;
