const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));
const softplus = (x: number) => Math.log(1 + Math.exp(x));
const clamp = (v: number, a: number, b: number) => Math.max(a, Math.min(b, v));
const nowMs = () => Date.now();
