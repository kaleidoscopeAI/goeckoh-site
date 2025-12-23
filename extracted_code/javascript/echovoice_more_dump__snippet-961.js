const na = this.nodes.get(edge.a)!, nb = this.nodes.get(edge.b)!;
const rij = na.neighbors.length + nb.neighbors.length - 2; // Discrete curvature
edge.w += -2 * rij * this.dt; // Smooth g ~ w
