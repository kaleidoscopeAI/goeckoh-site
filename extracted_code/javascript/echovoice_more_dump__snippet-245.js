for (const X of constructs) {
const raw = modulators[X];
const maturityFactor = sigmoid(maturityScalar); // [0,1]
const scaled = raw * clamp(1 - homeostasisPenalty[X], 0, 1) * (0.2 + 0.8*maturityFactor);
