export function updateNode(node: Node, e: EmotionalVector, dt = 0.01): Node {
const dK = (node.A * e.joy - node.R * e.fear) * dt;
const dA = (node.V * e.anticipation - node.C * e.disgust) * dt;
const dV = (node.K * e.trust - node.D * e.anger) * dt;
const dD = (node.K * e.sadness - node.A * e.joy) * dt;
const dC = (node.R * e.surprise - node.D * e.fear) * dt;
const dR = (node.C * e.trust - node.V * e.anger) * dt;
