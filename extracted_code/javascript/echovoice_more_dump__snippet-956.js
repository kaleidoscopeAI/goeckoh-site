const nb = this.nodes.get(neighId)!;
const oldSim = this.bitSim(n.e, nb.e);
n.e[bitK] = this.bitProbabilistic ? clip(oldVal + 0.5 * (Math.random() - 0.5), 0, 1) : 1 - oldVal; // Trial
const newSim = this.bitSim(n.e, nb.e);
delta += this.lambdaBit * ((1 - newSim) - (1 - oldSim));
n.e[bitK] = oldVal; // Reset
