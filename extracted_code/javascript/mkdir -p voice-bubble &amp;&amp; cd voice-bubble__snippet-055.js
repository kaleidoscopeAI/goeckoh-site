const K = N / 2;
const mags = new Float32Array(K);
for (let k = 1; k < K; k++) {
let re = 0, im = 0;
const ang = -2 * Math.PI * k / N;
for (let n = 0; n < N; n++) {
const xn = x[n] * w[n];
const a = ang * n;
