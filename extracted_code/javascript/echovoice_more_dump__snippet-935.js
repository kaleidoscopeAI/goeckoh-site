const grads = new Map<string, Vec3>();
for (const n of this.nodes.values()) grads.set(n.id, [0, 0, 0]);
for (const edge of this.edges) {
  const na = this.nodes.get(edge.a)!, nb = this.nodes.get(edge.b)!;
  const diff = sub(na.pos, nb.pos);
  const term = scale(diff, 2 * this.lambdaPos);
  grads.set(na.id, add(grads.get(na.id)!, term));
  grads.set(nb.id, sub(grads.get(nb.id)!, term));
}
for (const n of this.nodes.values()) {
  const tetherGrad = scale(sub(n.pos, [0, 0, 0]), 2 * this.alphaTether);
  grads.set(n.id, add(grads.get(n.id)!, tetherGrad));
}
return grads;
