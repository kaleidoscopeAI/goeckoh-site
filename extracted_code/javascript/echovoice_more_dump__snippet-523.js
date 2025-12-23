const CovCopy: number[][] = Cov.map(row => row.slice());
const e1 = powerIterationSymmetric(CovCopy, 500, 1e-9);
