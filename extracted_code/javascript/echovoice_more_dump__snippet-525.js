const coords: { x: number; y: number }[] = [];
const sqrt1 = Math.sqrt(Math.max(0, e1.eigenvalue));
const sqrt2 = Math.sqrt(Math.max(0, e2.eigenvalue));
for (let j = 0; j < this.M; j++) {
const x = (e1.eigenvector[j] ?? 0) * sqrt1;
const y = (e2.eigenvector[j] ?? 0) * sqrt2;
