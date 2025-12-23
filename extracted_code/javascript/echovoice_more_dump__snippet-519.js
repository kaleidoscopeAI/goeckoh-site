const C: number[] = new Array(this.M).fill(0);
for (let j = 0; j < this.M; j++) {
let s = 0;
for (let i = 0; i < N; i++) s += Ki[i] * (this.W![i][j] ?? 0);
