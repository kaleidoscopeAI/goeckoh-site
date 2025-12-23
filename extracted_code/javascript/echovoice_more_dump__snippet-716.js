const summaries = [];
for (let j = 0; j < M; j++) {
const contributions = [];
let activation = 0;
for (let i = 0; i < N; i++) {
const ki = Number(nodes[i]?.Ki ?? 0);
const contrib = (W[i][j] ?? 0) * ki;
