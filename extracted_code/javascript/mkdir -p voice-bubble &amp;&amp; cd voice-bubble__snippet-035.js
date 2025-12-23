const rms = Math.sqrt(sumSq / frame.length);
const energy = clamp(rms * 3.2, 0, 1); // scale into 0..1

const zcr = clamp(zc / frame.length, 0, 1);

