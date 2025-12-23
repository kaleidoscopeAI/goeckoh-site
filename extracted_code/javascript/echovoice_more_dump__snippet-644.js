console.log(`Running trial seed=${seed} lr=${lr} clip=${clip} applyEvery=${applyEvery}`);
const svc = new ProjectionService({ seed, learner: { enabled: true, applyEvery }, persistenceDecay: 0.94 });
