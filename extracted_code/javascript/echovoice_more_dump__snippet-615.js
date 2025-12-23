for (let a = 0; a < M; a++) {
for (let b = a; b < M; b++) {
let s = 0;
for (let i = 0; i < N; i++) s += V[a][i] * V[b][i];
const val = s / (N - 1);
