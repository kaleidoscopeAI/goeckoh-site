const na = engine.nodes.get(edge.a)!, nb = engine.nodes.get(edge.b)!;
const overlap = na.neighbors.filter(id => nb.neighbors.includes(id)).length; // Common neighbors
const rij = na.neighbors.length + nb.neighbors.length - 2 - overlap; // Discrete Ricci
edge.w -= 2 * rij * engine.dt; // Evolve metric g ~ w
edge.w = Math.max(0.1, edge.w); // Clamp for stability
