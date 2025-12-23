export function computeModulators(node: NodeState): Modulators {
const s = node.species;
const selfRaw = projectVec(s, DEFAULT_P.self);
const selfDelta = sigmoid(selfRaw) * 2 - 1;
const selfConfGain = clamp(selfRaw * 0.5 + 0.5, 0, 2);
const reflRaw = projectVec(s, DEFAULT_P.reflection);
const reflectionProb = clamp(
