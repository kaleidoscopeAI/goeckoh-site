const neighbors: number[] = [];
const cellRadius = Math.ceil(radius / this.cellSize);
const baseX = Math.floor(x / this.cellSize);
const baseY = Math.floor(y / this.cellSize);
const baseZ = Math.floor(z / this.cellSize);

for (let dx = -cellRadius; dx <= cellRadius; dx++) {
  for (let dy = -cellRadius; dy <= cellRadius; dy++) {
    for (let dz = -cellRadius; dz <= cellRadius; dz++) {
      const key = `${baseX+dx},${baseY+dy},${baseZ+dz}`;
      if (this.grid.has(key)) {
        neighbors.push(...this.grid.get(key)!);
      }
    }
  }
}
return neighbors;
