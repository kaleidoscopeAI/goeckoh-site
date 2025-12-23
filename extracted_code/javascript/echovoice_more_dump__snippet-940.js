const n = this.nodes.get(nodeId)!;
let delta = 0;
const oldE = n.e[bitK];
n.e[bitK] = this.bitProbabilistic ? 0.5 : 1 - oldE; // Temp flip
for (const neighId of n.neighbors) {
  const edge = this.edges.find(e => (e.a === nodeId && e.b === neighId) || (e.b === nodeId && e.a === neighId))!;
  const nb = this.nodes.get(neighId)!;
  const oldSim = this.bitSim(n.e, nb.e); // With flip? No, compute diff
  n.e[bitK] = oldE;
  const newSim = this.bitSim(n.e, nb.e);
  n.e[bitK] = 1 - oldE; // Restore for calc
  delta += this.lambdaBit * ((1 - newSim) - (1 - oldSim));
}
n.e[bitK] = oldE; // Reset
return delta;
