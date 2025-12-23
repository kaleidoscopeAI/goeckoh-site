if (Math.random() > n.repProb) return;
const newPos = add(n.pos, scale([gaussian(), gaussian(), gaussian()], 0.5));
const newE = n.e.map(v => this.bitProbabilistic ? clip(v + gaussian() * 0.05, 0, 1) : v);
const newN = this.addNode(newPos, newE);
newN.b = n.b + gaussian() * 0.05;
newN.h = n.h + gaussian() * 0.05;
// etc for others
for (const neigh of n.neighbors.slice(0, Math.floor(n.neighbors.length / 2))) {
  this.addEdge(newN.id, neigh);
}
