const reasonRaw = projectVec(s, DEFAULT_P.reason);
const attentionGain = clamp(1 + 0.5 * reasonRaw, 0.5, 3.0);
const plannerDepthFactor = clamp( Math.floor(1 + 2 * sigmoid(reasonRaw)), 1, 3 ); // integer depth delta
const plannerTempFactor = clamp(Math.exp(0.3 * reasonRaw), 0.5, 3.0);
