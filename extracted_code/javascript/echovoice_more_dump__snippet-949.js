this.edges = this.edges.filter(e => !(e.a === a && e.b === b || e.a === b && e.b === a));
const na = this.nodes.get(a)!, nb = this.nodes.get(b)!;
na.neighbors = na.neighbors.filter(id => id !== b);
nb.neighbors = nb.neighbors.filter(id => id !== a);
