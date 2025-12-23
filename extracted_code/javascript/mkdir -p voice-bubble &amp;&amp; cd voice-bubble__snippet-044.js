const t0 = tauEstimate;
const tPrev = Math.max(t0 - 1, tauMin);
const tNext = Math.min(t0 + 1, tauMax);
const s0 = cmnd[t0], s1 = cmnd[tPrev], s2 = cmnd[tNext];
const denom = (2 * s0 - s1 - s2);
let betterTau = t0;
