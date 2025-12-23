  const states: number[][] = [];
  for (const n of this.nodes.values()) {
    states.push([...n.pos, n.b, n.h, n.kappa, n.mu]); // 7D
  }
  if (states.length < 2) return 0;
  const cov = computeCovariance(states);
  const mid = Math.floor(states.length / 2);
  const covA = computeCovariance(states.slice(0, mid));
  const covB = computeCovariance(states.slice(mid));
  const d = det(cov) + 1e-10;
  const dA = det(covA) + 1e-10;
  const dB = det(covB) + 1e-10;
  return 0.5 * Math.log(d / (dA * dB));
