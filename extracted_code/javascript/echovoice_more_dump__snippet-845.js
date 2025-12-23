export function tuneDamping(node: Node, e: EmotionalVector, baseDt = 0.01): number {
const rho = spectralRadius(node, e);
