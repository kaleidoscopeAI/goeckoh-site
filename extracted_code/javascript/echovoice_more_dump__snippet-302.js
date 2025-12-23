const out = new Float32Array(n * m);
for (let i = 0; i < n; i++) {
for (let j = 0; j < m; j++) out[i * m + j] = this.S[i * m + j];
