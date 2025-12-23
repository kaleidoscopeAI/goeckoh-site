function outer(a: number[], b: number[]): number[][] {
  return a.map(ai => b.map(bj => ai * bj));
