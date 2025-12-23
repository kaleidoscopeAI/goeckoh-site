export function computeCovariance(data: number[][]): number[][] {
  if (data.length < 2) return data[0].map(() => data[0].map(() => 0));
  const dim = data[0].length;
  const mean = new Array(dim).fill(0);
  data.forEach(row => row.forEach((v, j) => mean[j] += v / data.length));
  let cov = new Array(dim).fill(0).map(() => new Array(dim).fill(0));
  data.forEach(row => {
    const dev = row.map((v, j) => v - mean[j]);
    cov = matrixAdd(cov, outer(dev, dev));
  });
  return matrixScale(cov, 1 / (data.length - 1));
