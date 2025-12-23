function rk4Step(f: (y: number[]) => number[], y: number[], dt: number): number[] {
  const k1 = f(y);
  const y2 = y.map((v, i) => v + dt * 0.5 * k1[i]);
  const k2 = f(y2);
  const y3 = y.map((v, i) => v + dt * 0.5 * k2[i]);
  const k3 = f(y3);
  const y4 = y.map((v, i) => v + dt * k3[i]);
  const k4 = f(y4);
  return y.map((v, i) => v + (dt / 6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]));
