export function det(m: number[][]): number {
  const n = m.length;
  if (n === 1) return m[0][0];
  if (n === 2) return m[0][0] * m[1][1] - m[0][1] * m[1][0];
  let sum = 0;
  for (let i = 0; i < n; i++) {
    const sub = m.map(row => row.slice(0, i).concat(row.slice(i + 1)));
    const sign = i % 2 === 0 ? 1 : -1;
    sum += sign * m[0][i] * det(sub.slice(1));
  }
  return sum;
