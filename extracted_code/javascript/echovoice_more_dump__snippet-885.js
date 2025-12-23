+export function flatten2D(mat: number[][]): Float64Array {
+ const m = mat.length;
+ const block = mat[0].length;
Printed using ChatGPT to PDF, powered by PDFCrowd HTML to PDF API.
136/141+ const out = new Float64Array(m * block);
+ for (let i = 0; i < m; i++) {
