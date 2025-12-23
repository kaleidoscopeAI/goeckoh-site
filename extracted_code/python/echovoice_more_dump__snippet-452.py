// Build Laplacian from adjacency (rows: neighbors with weights)
export function buildLaplacian(adj: SparseMatrix): SparseMatrix {
const n = adj.n;
const rows = new Array(n);
for (let i = 0; i < n; i++) {
const r = adj.rows[i];
