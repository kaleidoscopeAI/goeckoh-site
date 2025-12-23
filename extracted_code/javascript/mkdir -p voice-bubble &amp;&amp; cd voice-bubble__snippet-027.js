const sr = audioCtx.sampleRate;
const win = 2048;
const hop = Math.max(1, Math.floor(sr * 0.01));
let buf = new Float32Array(win);
let writeIdx = 0;
let samplesSince = 0;

