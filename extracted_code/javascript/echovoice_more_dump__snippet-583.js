const dists = constructs.map((b, idx) => ({ idx, d: Math.hypot(a.coord.x - b.coord.x, a.coord.y - b.coord.y) }));
