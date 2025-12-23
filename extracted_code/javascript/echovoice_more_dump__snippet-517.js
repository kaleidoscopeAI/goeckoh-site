function l2Normalize(v: EVector): EVector {
const out = copyVector(v);
const sumSq = Math.sqrt(Object.values(out).reduce((s, x) => s + x * x, 0) || 1);
for (const k of Object.keys(out)) out[k as EmotionName] = out[k as EmotionName] / sumSq;
