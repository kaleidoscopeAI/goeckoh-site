// Approx with positions + engines as states
const states: number[][] = [];
for (const n of this.nodes.values()) states.push([...n.pos, n.b, n.h, n.kappa, n.mu]);
const cov = this.computeCovariance(states); // Implement covariance matrix
// Min over balanced partitions (heuristic: half)
const mid = Math.floor(states.length / 2);
const covA = this.computeCovariance(states.slice(0, mid));
const covB = this.computeCovariance(states.slice(mid));
const det = this.det(cov); // Implement det or use lib
const detA = this.det(covA);
const detB = this.det(covB);
return 0.5 * Math.log(det / (detA * detB + 1e-10));
