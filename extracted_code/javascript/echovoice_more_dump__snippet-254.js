const selfDelta = sigmoid(selfRaw) * 2 - 1; // map to [-1,1]
const selfConfGain = clamp(selfRaw * 0.5 + 0.5, 0, 2);
