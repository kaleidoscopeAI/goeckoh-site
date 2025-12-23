const displayScale = maxAbs > 0 ? (1.0 / maxAbs) * 2.5 : 1.0; // scale to ~[-2.5,2.5]
for (let j = 0; j < this.M; j++) coords[j].x *= displayScale, coords[j].y *= displayScale;
