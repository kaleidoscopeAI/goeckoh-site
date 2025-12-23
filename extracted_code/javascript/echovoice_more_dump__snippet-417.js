function l1Normalize(v: EVector): EVector {
const out = copyVector(v);
const sum = Object.values(out).reduce((s, x) => s + Math.abs(x), 0) || 1;
for (const k of Object.keys(out)) out[k as EmotionName] = out[k as EmotionName] / sum;
