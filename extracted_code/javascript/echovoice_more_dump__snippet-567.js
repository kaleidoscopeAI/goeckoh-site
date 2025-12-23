const e2 = powerIterationSymmetric(CovCopy, 500, 1e-9);
const sqrt1 = Math.sqrt(Math.max(0, e1.eigenvalue));
const sqrt2 = Math.sqrt(Math.max(0, e2.eigenvalue));
const coords = [];
for (let j = 0; j < this.M; j++) coords.push({ x: (e1.eigenvector[j] ?? 0) * sqrt1, y: (e2.eigenvector[j] ?? 0) * sqrt2 });
