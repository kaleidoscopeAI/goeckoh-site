export function mul(mat: SparseMatrix, v: Float32Array): Float32Array {
const out = new Float32Array(mat.n);
for (let i = 0; i < mat.n; i++) {
const r = mat.rows[i];
let s = 0;
for (let k = 0; k < r.idx.length; k++) {
