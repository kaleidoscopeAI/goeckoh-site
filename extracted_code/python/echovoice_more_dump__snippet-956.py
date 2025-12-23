for (const n of this.nodes.values()) {
  const states = [n.b, n.h, n.kappa, n.mu];
  const f = (y: number[]) => {
    const rhs = new Array(4).fill(0);
    const neighDiff = [0, 0, 0, 0];
    for (const neighId of n.neighbors) {
      const nb = this.nodes.get(neighId)!;
      neighDiff[0] += nb.b - y[0];
      neighDiff[1] += nb.h - y[1];
      neighDiff[2] += nb.kappa - y[2];
      neighDiff[3] += nb.mu - y[3];
    }
    const numNeigh = n.neighbors.length || 1;
    // Aux: I_i approx from energy, o_i from a, etc.
    const iInput = Math.random() * 0.1; // From data/LLM
    const o = n.a;
    const epsilon = gaussian() * 0.05;
    const pSignal = n.b; // Cycle
    const sSignal = n.h;
    const eMeta = n.energy;
    const deltaErr = gaussian() * 0.1; // Mirror error
    const sigmaSmooth = 0.05;
    rhs[0] = this.engineAlpha[0] * iInput * o - this.engineBeta[0] * y[0] + this.engineGamma[0] * (neighDiff[0] / numNeigh) + gaussian() * 0.01;
    rhs[1] = this.engineAlpha[1] * (iInput + epsilon) - this.engineBeta[1] * y[1] + this.engineGamma[1] * (neighDiff[1] / numNeigh) + gaussian() * 0.01;
    rhs[2] = this.engineAlpha[2] * (pSignal + sSignal + eMeta) - this.engineBeta[2] * y[2] + this.engineGamma[2] * (neighDiff[2] / numNeigh) + gaussian() * 0.01;
    rhs[3] = -this.engineAlpha[3] * deltaErr + this.engineBeta[3] * sigmaSmooth + this.engineGamma[3] * (neighDiff[3] / numNeigh) + gaussian() * 0.01;
    return rhs;
  };
  const newStates = rk4Step(f, states, this.dt);
  n.b = newStates[0];
  n.h = newStates[1];
  n.kappa = newStates[2];
  n.mu = newStates[3];
}
