const e: Edge = { a, b, k: k ?? this.kBondDefault, l: l ?? this.lBondDefault, w, j };
this.edges.push(e);
const na = this.nodes.get(a)!, nb = this.nodes.get(b)!;
na.neighbors.push(b);
nb.neighbors.push(a);
