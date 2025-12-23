this.grid.clear();
for (let i = 0; i < count; i++) {
  const i3 = i * 3;
  const key = this.hash(positions[i3], positions[i3+1], positions[i3+2]);
  if (!this.grid.has(key)) this.grid.set(key, []);
  this.grid.get(key)!.push(i);
}
