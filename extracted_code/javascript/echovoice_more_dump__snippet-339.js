export function buildLaplacian(adj: SparseMatrix): SparseMatrix {
const n = adj.n;
const rows = new Array(n);
for (let i = 0; i < n; i++) {
const r = adj.rows[i];
let deg = 0;
for (let w of r.vals) deg += w;
const idx: number[] = [];
Printed using ChatGPT to PDF, powered by PDFCrowd HTML to PDF API.
