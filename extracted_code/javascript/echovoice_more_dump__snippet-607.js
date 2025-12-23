const svc = new ProjectionService({ constructs: ["A","B","C"], seed: 42 });
const constructs = svc.update(nodes);
