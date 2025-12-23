// Energy + ZCR
let sumSq = 0;
let zc = 0;
let prev = frame[0];
for (let i = 0; i < frame.length; i++) {
  const x = frame[i];
  sumSq += x * x;
  if ((x >= 0 && prev < 0) || (x < 0 && prev >= 0)) zc++;
  prev = x;
}
const rms = Math.sqrt(sumSq / frame.length);
const energy = clamp(rms * 3.2, 0, 1);
const zcr = clamp(zc / frame.length, 0, 1);

// Pitch via YIN
const f0 = this._yinPitch(frame, 80, 400);

// HNR estimate
const hnr = this._estimateHNR(frame, f0);

// Spectral tilt
const tilt = this._spectralTilt(frame);

return { energy, f0, zcr, hnr, tilt };
}

