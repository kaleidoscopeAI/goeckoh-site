const f0 = yinPitch(frame, sr, 80, 400);

// HNR (dB) using autocorrelation peak ratio at pitch lag
const hnr = estimateHNR(frame, sr, f0);

