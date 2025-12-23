const svc = new ProjectionService({ seed: 42, learner: { enabled: true, applyEvery: 1 }});
const nodes = [];
for (let i = 0; i < 20; i++) nodes.push({ id: i, Ki: Math.random() * 0.2 - 0.1});
const constructsBefore = svc.update(nodes);
