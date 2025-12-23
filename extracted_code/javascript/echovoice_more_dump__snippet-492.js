function colorMap(value: number) {
const r = Math.min(1, value * 2);
const g = Math.min(1, (1 - value) * 2);
const b = Math.min(1, 0.5 + value * 0.5);
