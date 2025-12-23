const n = this.graph.n, m = this.cfg.speciesCount;
const k = this.cfg.projectP.length / m;
const E = new Float32Array(n * k);
for (let i = 0; i < n; i++) {
for (let ki = 0; ki < k; ki++) {
let sum = 0;
for (let j = 0; j < m; j++) sum += this.cfg.projectP[ki * m + j] * this.S[i * m + j];
