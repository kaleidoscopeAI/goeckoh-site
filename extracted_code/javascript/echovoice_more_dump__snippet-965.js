const remote = JSON.parse(snap);
const deltaPhi = Math.cos(engine.avgB() - remote.avgB); // Phase coherence
if (deltaPhi > 0.8) engine.averageStates(remote); // Sync if coherent
