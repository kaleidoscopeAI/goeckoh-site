export class Engine {
  nodes: Map<string, NodeState> = new Map();
  edges: Edge[] = [];
  dt = 0.001; // Small as per spec
  damping = 0.9;
  kBondDefault = 8.0;
  lBondDefault = 1.6;
  lambdaBit = 1.0; // Defaults from spec
  lambdaPos = 0.5;
  alphaTether = 0.01;
  bitProbabilistic = false; // Toggle for p_k vs binary
  dBit = 128;
  etaSem = 0.18;
  mutationSigmaDefault = 0.02;
  llmEta = 0.06;
  replicateLambda = 8.0;
  knowledgeDecay = 0.995;
  thetaK = 0.4;
  thetaA = 0.2;
  thetaE = 0.1;
  maxNodes = 1000;
  t = 0; // Time
  temperature = 1.0; // For Metropolis
  annealRate = 0.999;
  emotionalSpecies = 1; // Num C species
  emotionalP = 0.05; // Production
  emotionalD = 0.1; // Decay
  emotionalGamma = 0.01; // Modulation strength
  engineAlpha = [0.05, 0.05, 0.05, 0.05]; // p,s,k,m
  engineBeta = [0.1, 0.1, 0.1, 0.1];
  engineGamma = [0.01, 0.01, 0.01, 0.01];
  birthPeriod = 50;
  sleepPeriod = 100;
  crystalEpsX = 1e-6;
  crystalEpsG = 1e-4;
  crystalEpsB = 0.01;
  crystalWindow = 500;
  wMin = 1e-3;
  history: { pos: Vec3; grad: number }[][] = []; // For crystallization

  constructor() {
    this.history = Array.from({ length: this.maxNodes }, () => []);
  }

  addNode(pos?: Vec3, e?: number[]): NodeState {
    if (this.nodes.size >= this.maxNodes) throw new Error("Max nodes");
    const id = randomUUID();
    const p: Vec3 = pos ?? [(Math.random() - 0.5) * 6, (Math.random() - 0.5) * 1, (Math.random() - 0.5) * 6];
    const bits = e ?? new Array(this.dBit).fill(0).map(() => Math.random() > 0.5 ? 1 : 0);
    const n: NodeState = {
      id, pos: p, vel: [0, 0, 0], mass: 1, e: bits, energy: 0, k: 0.5, a: 0.2,
      b: gaussian() * 0.1, h: gaussian() * 0.1, kappa: gaussian() * 0.1, mu: gaussian() * 0.1,
      c: new Array(this.emotionalSpecies).fill(0.5), s: Math.random() > 0.5 ? 1 : -1,
      mutation_sigma: this.mutationSigmaDefault, repProb: 0, pruneRisk: 0, neighbors: []
    };
    this.nodes.set(id, n);
    this.history.push([]);
    return n;
  }

  addEdge(a: string, b: string, k?: number, l?: number, w = 1.0, j = 1.0) {
    const e: Edge = { a, b, k: k ?? this.kBondDefault, l: l ?? this.lBondDefault, w, j };
    this.edges.push(e);
    const na = this.nodes.get(a)!, nb = this.nodes.get(b)!;
    na.neighbors.push(b);
    nb.neighbors.push(a);
  }

  // Bit similarity (Hamming for binary, prob for soft)
  bitSim(ei: number[], ej: number[]): number {
    if (!this.bitProbabilistic) {
      let diff = 0;
      for (let k = 0; k < this.dBit; k++) diff += ei[k] !== ej[k] ? 1 : 0;
      return 1 - diff / this.dBit;
    } else {
      let sum = 0;
      for (let k = 0; k < this.dBit; k++) {
        sum += ei[k] * ej[k] * (2 * (ei[k] === ej[k] ? 1 : 0) - 1); // Approx delta
      }
      return sum / this.dBit;
    }
  }

  // Full Hamiltonian H
  computeH(): number {
    let h = 0;
    for (const edge of this.edges) {
      const na = this.nodes.get(edge.a)!, nb = this.nodes.get(edge.b)!;
      const simBit = this.bitSim(na.e, nb.e);
      const distSq = len(sub(na.pos, nb.pos)) ** 2;
      const attr = edge.w * (-edge.j * na.s * nb.s); // Spin-glass optional
      h += this.lambdaBit * (1 - simBit) + this.lambdaPos * distSq + attr;
    }
    for (const n of this.nodes.values()) {
      const tether = this.alphaTether * len(sub(n.pos, [0, 0, 0])) ** 2; // x_i0 = 0 for simplicity
      h += tether; // + reg terms if needed
    }
    return h;
  }

  // Grad x_i H
  computePosGrads(): Map<string, Vec3> {
    const grads = new Map<string, Vec3>();
    for (const n of this.nodes.values()) grads.set(n.id, [0, 0, 0]);
    for (const edge of this.edges) {
      const na = this.nodes.get(edge.a)!, nb = this.nodes.get(edge.b)!;
      const diff = sub(na.pos, nb.pos);
      const term = scale(diff, 2 * this.lambdaPos);
      grads.set(na.id, add(grads.get(na.id)!, term));
      grads.set(nb.id, sub(grads.get(nb.id)!, term));
    }
    for (const n of this.nodes.values()) {
      const tetherGrad = scale(sub(n.pos, [0, 0, 0]), 2 * this.alphaTether);
      grads.set(n.id, add(grads.get(n.id)!, tetherGrad));
    }
    return grads;
  }

  // Local energy for node (PDF viz)
  computeLocalEnergies() {
    for (const n of this.nodes.values()) {
      let localH = 0;
      for (const neighId of n.neighbors) {
        const edge = this.edges.find(e => (e.a === n.id && e.b === neighId) || (e.b === n.id && e.a === neighId))!;
        const nb = this.nodes.get(neighId)!;
        const simBit = this.bitSim(n.e, nb.e);
        const distSq = len(sub(n.pos, nb.pos)) ** 2;
        const attr = edge.w * (-edge.j * n.s * nb.s);
        localH += 0.5 * (this.lambdaBit * (1 - simBit) + this.lambdaPos * distSq + attr);
      }
      localH += this.alphaTether * len(sub(n.pos, [0, 0, 0])) ** 2;
      n.energy = localH;
    }
  }

  // Step positions with leapfrog (symplectic)
  stepPhysics() {
    const grads1 = this.computePosGrads();
    for (const n of this.nodes.values()) {
      const g1 = grads1.get(n.id)!;
      const acc1 = scale(g1, -1 / n.mass); // -grad for descent
      n.vel = add(n.vel, scale(acc1, this.dt / 2));
      n.pos = add(n.pos, scale(n.vel, this.dt));
    }
    const grads2 = this.computePosGrads();
    for (const n of this.nodes.values()) {
      const g2 = grads2.get(n.id)!;
      const acc2 = scale(g2, -1 / n.mass);
      n.vel = add(n.vel, scale(acc2, this.dt / 2));
      n.vel = scale(n.vel, this.damping);
      n.vel = add(n.vel, [gaussian() * 0.01, gaussian() * 0.01, gaussian() * 0.01]); // Noise
    }
  }

  // Semantic diffusion + mutation (PDF)
  semanticStep() {
    const newE = new Map<string, number[]>();
    for (const n of this.nodes.values()) newE.set(n.id, n.e.slice());
    for (const n of this.nodes.values()) {
      const accum = new Array(this.dBit).fill(0);
      const numNeigh = n.neighbors.length || 1;
      for (const neighId of n.neighbors) {
        const nb = this.nodes.get(neighId)!.e;
        for (let d = 0; d < this.dBit; d++) accum[d] += (nb[d] - n.e[d]) / numNeigh;
      }
      const out = newE.get(n.id)!;
      for (let d = 0; d < this.dBit; d++) {
        out[d] += this.etaSem * accum[d] + gaussian() * n.mutation_sigma;
        if (this.bitProbabilistic) out[d] = clip(out[d], 0, 1);
        else out[d] = Math.random() < out[d] ? 1 : 0; // Threshold if soft
      }
    }
    for (const [id, arr] of newE) this.nodes.get(id)!.e = arr;
  }

  // Bit updates: Metropolis flips (spec)
  bitStep() {
    for (const n of this.nodes.values()) {
      const k = Math.floor(Math.random() * this.dBit); // Random bit to flip
      const oldE = n.e[k];
      n.e[k] = this.bitProbabilistic ? clip(gaussian() * 0.1 + oldE, 0, 1) : 1 - oldE;
      const deltaH = this.computeDeltaHForBit(n.id, k); // Incremental
      const pAccept = Math.min(1, Math.exp(-deltaH / this.temperature));
      if (Math.random() > pAccept) n.e[k] = oldE; // Reject
    }
  }

  // Incremental ΔH for bit flip (only neighbors)
  computeDeltaHForBit(nodeId: string, bitK: number): number {
    const n = this.nodes.get(nodeId)!;
    let delta = 0;
    const oldE = n.e[bitK];
    n.e[bitK] = this.bitProbabilistic ? 0.5 : 1 - oldE; // Temp flip
    for (const neighId of n.neighbors) {
      const edge = this.edges.find(e => (e.a === nodeId && e.b === neighId) || (e.b === nodeId && e.a === neighId))!;
      const nb = this.nodes.get(neighId)!;
      const oldSim = this.bitSim(n.e, nb.e); // With flip? No, compute diff
      n.e[bitK] = oldE;
      const newSim = this.bitSim(n.e, nb.e);
      n.e[bitK] = 1 - oldE; // Restore for calc
      delta += this.lambdaBit * ((1 - newSim) - (1 - oldSim));
    }
    n.e[bitK] = oldE; // Reset
    return delta;
  }

  // Thought engines ODEs (RK4, spec)
  enginesStep() {
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
  }

  // Emotional flow ODEs (RK4, spec)
  emotionalStep() {
    for (const n of this.nodes.values()) {
      const f = (yc: number[]) => {
        const rhs = new Array(this.emotionalSpecies).fill(0);
        for (let sp = 0; sp < this.emotionalSpecies; sp++) {
          let transportIn = 0, transportOut = 0;
          for (const neighId of n.neighbors) {
            const nb = this.nodes.get(neighId)!;
            const tIj = 0.05; // Fixed transport rate
            transportIn += tIj * nb.c[sp];
            transportOut += tIj * yc[sp];
          }
          const sExt = gaussian() * 0.01; // Stimuli
          rhs[sp] = this.emotionalP - this.emotionalD * yc[sp] + transportIn - transportOut + sExt;
        }
        return rhs;
      };
      n.c = rk4Step(f, n.c, this.dt);
    }
    // Modulate lambdas
    const avgC = Array.from(this.nodes.values()).reduce((sum, n) => sum.map((v, i) => v + n.c[i]), new Array(this.emotionalSpecies).fill(0)).map(v => v / this.nodes.size);
    this.lambdaBit *= (1 + this.emotionalGamma * Math.tanh(avgC[0])); // Example f(C)
    this.lambdaPos *= (1 + this.emotionalGamma * Math.tanh(avgC[0])); // g(C)
  }

  // Compute awareness, knowledge, repProb, pruneRisk (PDF metrics + spec)
  computeMetrics() {
    this.computeLocalEnergies();
    const globalH = this.computeH();
    // Knowledge K: avg confidence (for prob) or low entropy
    for (const n of this.nodes.values()) {
      if (this.bitProbabilistic) {
        n.k = n.e.reduce((sum, p) => sum + p * (1 - p), 0) / this.dBit; // Entropy approx, low=high K
        n.k = 1 - n.k; // Invert to confidence
      } else n.k *= this.knowledgeDecay; // Decay as before
      // Awareness A: softmax over energy + engines
      n.a = Math.exp(-n.energy + n.b + n.h + n.kappa + n.mu);
    }
    const sumA = Array.from(this.nodes.values()).reduce((sum, n) => sum + n.a, 0) || 1;
    for (const n of this.nodes.values()) {
      n.a /= sumA;
      const ell = this.replicateLambda * ((n.k - this.thetaK) + (n.a - this.thetaA) - (n.energy / globalH - this.thetaE));
      n.repProb = 1 / (1 + Math.exp(-ell));
      n.pruneRisk = n.energy / globalH > 0.1 ? 0.5 : 0; // High energy risk
    }
  }

  // Integrated info Φ approx (Gaussian, spec)
  computeIIT(): number {
    // Approx with positions + engines as states
    const states: number[][] = [];
    for (const n of this.nodes.values()) states.push([...n.pos, n.b, n.h, n.kappa, n.mu]);
    const cov = this.computeCovariance(states); // Implement covariance matrix
    // Min over balanced partitions (heuristic: half)
    const mid = Math.floor(states.length / 2);
    const covA = this.computeCovariance(states.slice(0, mid));
    const covB = this.computeCovariance(states.slice(mid));
    const det = this.det(cov); // Implement det or use lib
    const detA = this.det(covA);
    const detB = this.det(covB);
    return 0.5 * Math.log(det / (detA * detB + 1e-10));
  }

  // Helper cov and det (simple 2D for demo, extend)
  computeCovariance(data: number[][]): number[][] {
    const mean = data[0].map((_, i) => data.reduce((sum, row) => sum + row[i], 0) / data.length);
    return mean.map(() => mean.map(() => 0)); // Placeholder, implement full
  }

  det(m: number[][]): number {
    // Implement for small dim
    return 1; // Placeholder
  }

  // Replication (PDF + spec birth)
  replicate(n: NodeState) {
    if (Math.random() > n.repProb) return;
    const newPos = add(n.pos, scale([gaussian(), gaussian(), gaussian()], 0.5));
    const newE = n.e.map(v => this.bitProbabilistic ? clip(v + gaussian() * 0.05, 0, 1) : v);
    const newN = this.addNode(newPos, newE);
    newN.b = n.b + gaussian() * 0.05;
    newN.h = n.h + gaussian() * 0.05;
    // etc for others
    for (const neigh of n.neighbors.slice(0, Math.floor(n.neighbors.length / 2))) {
      this.addEdge(newN.id, neigh);
    }
  }

  // Pruning/death (spec)
  prune() {
    for (const n of [...this.nodes.values()]) {
      if (Math.random() < n.pruneRisk) {
        this.removeNode(n.id);
      }
    }
    this.edges = this.edges.filter(e => this.nodes.has(e.a) && this.nodes.has(e.b));
  }

  removeNode(id: string) {
    this.nodes.delete(id);
    this.edges = this.edges.filter(e => e.a !== id && e.b !== id);
    for (const n of this.nodes.values()) {
      n.neighbors = n.neighbors.filter(neigh => neigh !== id);
    }
  }

  // Bond rewiring (spec, periodic)
  rewireBonds() {
    for (const edge of [...this.edges]) {
      if (edge.w < this.wMin) {
        this.removeEdge(edge.a, edge.b);
      } else {
        // Propose toggle w, accept if ΔH <0 or Metropolis
        const oldW = edge.w;
        edge.w = Math.random();
        const deltaH = this.computeH() - this.computeH(); // Full, optimize
        const p = Math.min(1, Math.exp(-deltaH / this.temperature));
        if (Math.random() > p) edge.w = oldW;
      }
    }
  }

  removeEdge(a: string, b: string) {
    this.edges = this.edges.filter(e => !(e.a === a && e.b === b || e.a === b && e.b === a));
    const na = this.nodes.get(a)!, nb = this.nodes.get(b)!;
    na.neighbors = na.neighbors.filter(id => id !== b);
    nb.neighbors = nb.neighbors.filter(id => id !== a);
  }

  // Crystallization (spec)
  crystallize() {
    for (const [idx, n] of [...this.nodes.values()].entries()) {
      const hist = this.history[idx];
      hist.push({ pos: n.pos, grad: len(this.computePosGrads().get(n.id)!) });
      if (hist.length > this.crystalWindow) hist.shift();
      if (hist.length === this.crystalWindow) {
        const meanPos = hist[0].pos.map((_, i) => hist.reduce((sum, h) => sum + h.pos[i], 0) / this.crystalWindow);
        const varX = hist.reduce((sum, h) => sum + len(sub(h.pos, meanPos)) ** 2, 0) / this.crystalWindow;
        const meanG = hist.reduce((sum, h) => sum + h.grad, 0) / this.crystalWindow;
        const bitEnt = this.bitProbabilistic ? -n.e.reduce((sum, p) => sum + p * Math.log(p + 1e-10) + (1 - p) * Math.log(1 - p + 1e-10), 0) / this.dBit : 0;
        if (varX < this.crystalEpsX && meanG < this.crystalEpsG && bitEnt < this.crystalEpsB) {
          this.alphaTether *= 10; // Freeze by strong tether
          // Persist if needed
        }
      }
    }
  }

  // Consolidation/sleep (prune, rewire, crystallize)
  consolidation() {
    this.prune();
    this.rewireBonds();
    this.crystallize();
  }

  step() {
    this.t += this.dt;
    this.emotionalStep(); // 1. Emotions RK4
    // 2. Modulate coeffs done in emotional
    this.stepPhysics(); // 3. Positions leapfrog
    this.enginesStep(); // 5. Engines RK4
    this.bitStep(); // 6. Bits Metropolis
    // 7. Quantum skip
    this.semanticStep(); // PDF diffusion
    this.computeMetrics(); // A, K, repProb, etc.
    for (const n of [...this.nodes.values()]) this.replicate(n);
    if (Math.floor(this.t / this.dt) % this.birthPeriod === 0) {
      // Propose new node
      const parent = [...this.nodes.values()][Math.floor(Math.random() * this.nodes.size)];
      const newPos = add(parent.pos, [gaussian(), gaussian(), gaussian()]);
      const newE = parent.e.map(v => v + gaussian() * 0.1);
      const tempN = this.addNode(newPos, newE);
      const deltaH = this.computeH() - this.computeH(); // Approx
      if (Math.random() > Math.min(1, Math.exp(-deltaH / this.temperature * 2))) this.removeNode(tempN.id); // Higher T_birth
    }
    if (Math.floor(this.t / this.dt) % this.sleepPeriod === 0) this.consolidation();
    this.temperature *= this.annealRate;
  }

  snapshot() {
    const phi = this.computeIIT();
    return {
      nodes: Array.from(this.nodes.values()).map(n => ({
        id: n.id, pos: n.pos, e_head: n.e.slice(0, 4), energy: n.energy, k: n.k, a: n.a, b: n.b, h: n.h,
        kappa: n.kappa, mu: n.mu, c: n.c, repProb: n.repProb, pruneRisk: n.pruneRisk
      })),
      edges: this.edges.map(e => ({ a: e.a, b: e.b, k: e.k, l: e.l, w: e.w })),
      stats: { totalH: this.computeH(), phi, totalEnergy: Array.from(this.nodes.values()).reduce((sum, n) => sum + n.energy, 0) }
    };
  }
