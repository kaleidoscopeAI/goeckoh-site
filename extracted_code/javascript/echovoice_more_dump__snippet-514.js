function clamp(v: number, a = CLAMP_MIN, b = CLAMP_MAX) { return Math.max(a, Math.min(b, v)); }
function makeZeroVector(emotions: EmotionName[]): EVector {
const out = {} as any;
for (const e of emotions) out[e] = 0;
