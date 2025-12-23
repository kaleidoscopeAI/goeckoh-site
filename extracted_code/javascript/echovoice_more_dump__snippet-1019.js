constructor() {
    this.core = new CognitiveCrystal({
        latticeSize: 3,
        annealRate: 0.02,
        noise: 0.1,
        decay: 0.05
    });
    this.timeStep = 0;
}

step(params = {}) {
    const { load = 0.5, noise = 0.1, decay = 0.05, externalData = null } = params;

    // feed forward into crystal annealing loop
    this.core.applyAnnealing({
        taskLoad: load,
        noiseLevel: noise,
        decayRate: decay,
        externalStimuli: externalData
    });

    this.timeStep++;
    return this.metrics();
}

metrics() {
    return {
        stress: this.core.stress(),
        energy: this.core.energy(),
        confidence: this.core.confidence(),
        harmony: this.core.harmony(),
        emergence: this.core.emergence(),
        memory: this.core.memorySnapshot()
    };
}
