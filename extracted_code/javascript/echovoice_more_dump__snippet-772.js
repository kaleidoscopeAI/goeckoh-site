const M = traces[0]?.W_before[0]?.length ?? 0;
const constructDelta: number[] = new Array(M).fill(0);
for (const tr of traces) {
const A = tr.activations ?? new Array(M).fill(1);
const before = tr.W_before;
const after = tr.W_after;
const N = before.length;
for (let j = 0; j < M; j++) {
let s = 0;
for (let i = 0; i < N; i++) s += Math.abs(after[i][j] - before[i][j]);
