const newPos = add(n.pos, scale([gaussian(), gaussian(), gaussian()], 0.5));
const newSem = n.sem.map(v => v + gaussian() * 0.05);
const newN = this.addNode(newPos, newSem);
for (const neigh of n.neighbors.slice(0, Math.floor(n.neighbors.length / 2))) {
  this.addEdge(newN.id, neigh);
}
