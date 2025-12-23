for (let p = 0; p < m; p++) {
const sp = this.S[base + p];
for (let q = 0; q < m; q++) {
const sq = this.S[base + q];
for (let k = 0; k < m; k++) {
const idx = k * m * m + p * m + q; // flattened
