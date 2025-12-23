const dev = row.map((v, j) => v - mean[j]);
cov = matrixAdd(cov, outer(dev, dev));
