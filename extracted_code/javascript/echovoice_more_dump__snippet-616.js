function powerIterationSymmetric(A: number[][], maxIter = 200, tol = 1e-6): { eigenvalue: number, eigenvector: number[] } {
const M = A.length;
let v = new Array(M).fill(0).map((_, i) => Math.random());
