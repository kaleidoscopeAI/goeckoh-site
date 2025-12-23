const Cov = computeCovarianceMatrix(V);
const CovCopy = Cov.map(r => r.slice());
const e1 = powerIterationSymmetric(CovCopy, 500, 1e-9);
