// Difference function d(tau)
// We allocate per call for correctness; still fast enough at hop=10ms.
const d = new Float32Array(tauMax + 1);
for (let tau = tauMin; tau <= tauMax; tau++) {
  let sum = 0;
  for (let i = 0; i < N - tau; i++) {
    const diff = x[i] - x[i + tau];
    sum += diff * diff;
  }
  d[tau] = sum;
}

// CMND
const cmnd = new Float32Array(tauMax + 1);
cmnd[0] = 1;
let runningSum = 0;
for (let tau = 1; tau <= tauMax; tau++) {
  runningSum += d[tau];
  cmnd[tau] = d[tau] * tau / (runningSum + 1e-12);
}

const threshold = 0.12;
let tauEstimate = -1;
for (let tau = tauMin; tau <= tauMax; tau++) {
  if (cmnd[tau] < threshold) {
    while (tau + 1 <= tauMax && cmnd[tau + 1] < cmnd[tau]) tau++;
    tauEstimate = tau;
    break;
  }
}
if (tauEstimate === -1) return 0;

// Parabolic interpolation around tauEstimate
const t0 = tauEstimate;
const tPrev = Math.max(t0 - 1, tauMin);
const tNext = Math.min(t0 + 1, tauMax);
const s0 = cmnd[t0], s1 = cmnd[tPrev], s2 = cmnd[tNext];
const denom = (2 * s0 - s1 - s2);

let betterTau = t0;
if (Math.abs(denom) > 1e-12) {
  betterTau = t0 + (s2 - s1) / (2 * denom);
}

const f0 = this.sr / betterTau;
if (!Number.isFinite(f0) || f0 < fMin || f0 > fMax) return 0;
return f0;
}

