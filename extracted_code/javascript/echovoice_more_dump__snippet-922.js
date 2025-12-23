const e = {a, b, k: k ?? this.k_bond_default, L: L ?? this.bond_rest_default};
this.edges.push(e);
const na = this.nodes.get(a), nb = this.nodes.get(b);
if (na && !na.neighbors.includes(b)) na.neighbors.push(b);
if (nb && !nb.neighbors.includes(a)) nb.neighbors.push(a);
