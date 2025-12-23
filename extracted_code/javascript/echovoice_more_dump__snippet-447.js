export function computeMetaGrad(node: any, logs: any[]): Record<string, FloatArr> {
const grad: Record<string, FloatArr> = {};
for (const k in DEFAULT_P) grad[k] = new Array(DEFAULT_P[k].length).fill(0);
const totalRegret = logs.reduce((sum, l) => sum + (l.regret ?? 0), 0);
