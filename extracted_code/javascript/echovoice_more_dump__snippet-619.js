const newLambda = w.reduce((s, x, i) => s + x * v[i], 0);
const wnorm = Math.hypot(...w) || 1;
const vNext = w.map(x => x / wnorm);
