const r = smoothstep(0.0, 0.5, x);
const g = smoothstep(0.2, 0.9, 1 - Math.abs(x - 0.55));
const b = smoothstep(0.35, 1.0, 1 - x);
