let maxAbs = 0;
for (const p of coords) maxAbs = Math.max(maxAbs, Math.abs(p.x), Math.abs(p.y));
const displayScale = maxAbs > 0 ? (1.0 / maxAbs) * 2.5 : 1.0;
for (let j = 0; j < this.M; j++) coords[j].x *= displayScale, coords[j].y *= displayScale;
