function yinPitch(x, sr, fMin, fMax) {
const N = x.length;
const tauMin = Math.floor(sr / fMax);
const tauMax = Math.floor(sr / fMin);
