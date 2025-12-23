// Generates gradient based on regret/emotion correlation from recent reflections
export function computeMetaGrad(node: NodeState, logs: any[]): Record<string, FloatArr> {
const grad: Record<string, FloatArr> = {};
for (const k in DEFAULT_P) grad[k] = new Array(DEFAULT_P[k].length).fill(0);
let totalRegret = 0;
for (const t of logs) totalRegret += t.regret ?? 0.0;
