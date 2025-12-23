this.nodes.delete(id);
this.edges = this.edges.filter(e => e.a !== id && e.b !== id);
for (const n of this.nodes.values()) {
  n.neighbors = n.neighbors.filter(neigh => neigh !== id);
}
