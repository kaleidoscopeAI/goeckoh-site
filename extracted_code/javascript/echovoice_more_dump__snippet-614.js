function computeCovarianceMatrix(V: number[][]): number[][] {
const M = V.length;
const N = V[0]?.length ?? 0;
const Cov = Array.from({ length: M }, () => new Array(M).fill(0));
