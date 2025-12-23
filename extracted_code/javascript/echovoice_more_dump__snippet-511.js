let lambda = 0;
for (let iter = 0; iter < maxIter; iter++) {
const w = new Array(M).fill(0);
for (let i = 0; i < M; i++) {
let sum = 0;
for (let j = 0; j < M; j++) sum += A[i][j] * v[j];
