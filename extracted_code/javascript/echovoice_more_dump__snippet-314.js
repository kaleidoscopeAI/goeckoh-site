const adj = makeSimpleGraph(3);
const graph = { n: 3, adj, L: undefined as any };
const cfg = { dt: 0.05, speciesCount: 12, diffusion: new Float32Array(12).fill(0.1) };
const svc = new ActuationService(graph as any, cfg as any);
