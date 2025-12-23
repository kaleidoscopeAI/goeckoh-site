function matrixScale(m: number[][], s: number): number[][] {
  return m.map(row => row.map(v => v * s));
