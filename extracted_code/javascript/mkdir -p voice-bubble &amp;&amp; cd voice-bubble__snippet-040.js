for (let tau = tauMin; tau <= tauMax; tau++) {
let sum = 0;
for (let i = 0; i < N - tau; i++) {
const diff = x[i] - x[i + tau];
