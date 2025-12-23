let sx = 0, sy = 0, sxx = 0, sxy = 0;
let count = 0;

// Only bins in ~80..5000 Hz
const kMin = Math.max(2, Math.floor(80 * N / sr));
const kMax = Math.min((N / 2) - 1, Math.floor(5000 * N / sr));

for (let k = kMin; k <= kMax; k++) {
  // DFT at bin k (real/imag)
  let re = 0, im = 0;
  const ang = -2 * Math.PI * k / N;
  // A small speed trick: step angle accum
  let cosA = 1, sinA = 0;
  const cosStep = Math.cos(ang);
  const sinStep = Math.sin(ang);
  for (let n = 0; n < N; n++) {
    const xn = x[n] * this.hann[n];
    re += xn * cosA;
    im += xn * sinA;
    // rotate
    const nc = cosA * cosStep - sinA * sinStep;
    const ns = sinA * cosStep + cosA * sinStep;
    cosA = nc; sinA = ns;
  }
  const mag = Math.sqrt(re * re + im * im) + 1e-12;

  const f = (k * sr) / N;
  const X = Math.log(f);
  const Y = Math.log(mag);

  sx += X; sy += Y; sxx += X * X; sxy += X * Y;
  count++;
}

if (count < 10) return 0;
const denom = (count * sxx - sx * sx) + 1e-12;
const slope = (count * sxy - sx * sy) / denom;
return clamp(slope, -2, 2);
}
