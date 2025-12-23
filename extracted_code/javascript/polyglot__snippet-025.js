const n = vec.length;
const parts = Math.max(1, Math.floor(n / 2));
const sysEnt = this.entropy(vec);
let partEnt = 0.0;
for (let i = 0; i < parts; i++) {
const subset = [];
for (let j = i; j < n; j += parts) subset.push(vec[j]);
