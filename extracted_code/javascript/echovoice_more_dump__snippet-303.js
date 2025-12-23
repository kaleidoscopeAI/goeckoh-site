export function applyRegulatoryFilter(node: NodeState, mods: Modulators, valueLayerOk = true): Modulators {
const s = node.species;
const h = node.homeostasis;
let homeoPenalty = 0;
for (let i = 0; i < s.length; i++) {
const ratio = (h[i]) / (0.0001 + Math.abs(s[i]) + 1e-6);
