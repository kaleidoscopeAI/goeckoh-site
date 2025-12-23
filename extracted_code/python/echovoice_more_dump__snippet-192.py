import { NodeState, Edge, Vec3 } from "./types";
import { randomUUID } from "crypto";

function add(a: Vec3, b: Vec3): Vec3 { return [a[0]+b[0], a[1]+b[1], a[2]+b[2]]; }
function sub(a: Vec3, b: Vec3): Vec3 { return [a[0]-b[0], a[1]-b[1], a[2]-b[2]]; }
function scale(a: Vec3, s: number): Vec3 { return [a[0]*s, a[1]*s, a[2]*s]; }
function len(a: Vec3) { return Math.sqrt(a[0]*a[0]+a[1]*a[1]+a[2]*a[2]); }
function normalize(a: Vec3) { const L = len(a)||1; return [a[0]/L, a[1]/L, a[2]/L]; }
function zeros(): Vec3 { return [0,0,0]; }
function gaussian(): number { return Math.random() * 2 - 1; } // Approx for noise

export class Engine {
  nodes: Map<string, NodeState> = new Map();
  edges: Edge[] = [];
  dt = 0.016;
  damping = 0.9;
  k_bond_default = 8.0;
  bond_rest_default = 1.6;
  D = 16; // Filled placeholder
  eta_sem = 0.18;
  mutation_sigma_default = 0.02;
  llm_eta = 0.06;
  replicate_lambda = 8.0;
  knowledge_decay = 0.995;
  theta_K = 0.4; // Thresholds for replication
  theta_A = 0.2;
  theta_E = 0.1;
  max_nodes = 1000; // Safety limit

  constructor() {}

  addNode(pos?: Vec3, sem?: number[]): NodeState {
    if (this.nodes.size >= this.max_nodes) throw new Error("Max nodes reached");
    const id = randomUUID();
    const p: Vec3 = pos ?? [(Math.random()-0.5)*6, (Math.random()-0.5)*1, (Math.random()-0.5)*6];
    const s = sem ?? new Array(this.D).fill(0).map(gaussian);
    const n: NodeState = {
      id, pos: p, vel: [0,0,0], mass: 1, energy: 0, K: Math.random()*0.5+0.1, A: 0.2,
      sem: s, mutation_sigma: this.mutation_sigma_default, repProb: 0, neighbors: []
    };
    this.nodes.set(id, n);
    return n;
  }

  addEdge(a: string, b: string, k?: number, L?: number) {
    const e = {a, b, k: k ?? this.k_bond_default, L: L ?? this.bond_rest_default};
    this.edges.push(e);
    const na = this.nodes.get(a), nb = this.nodes.get(b);
    if (na && !na.neighbors.includes(b)) na.neighbors.push(b);
    if (nb && !nb.neighbors.includes(a)) nb.neighbors.push(a);
  }

  computeForces(): Map<string, Vec3> {
    const forces = new Map<string, Vec3>();
    for (const [id, n] of this.nodes) forces.set(id, [0, -0.01, 0]); // Gravity
    for (const e of this.edges) {
      const na = this.nodes.get(e.a), nb = this.nodes.get(e.b);
      if (!na || !nb) continue;
      const r = sub(na.pos, nb.pos);
      const dist = len(r) + 1e-9;
      const u = normalize(r);
      const mag = -e.k * (dist - e.L);
      const fij = scale(u, mag);
      forces.set(na.id, add(forces.get(na.id)!, fij));
      forces.set(nb.id, sub(forces.get(nb.id)!, fij));
    }
    return forces;
  }

  stepPhysics() {
    const F = this.computeForces();
    for (const [id, n] of this.nodes) {
      const Fi = F.get(id)!;
      const acc = scale(Fi, 1 / n.mass);
      n.pos = add(n.pos, add(scale(n.vel, this.dt), scale(acc, 0.5 * this.dt * this.dt)));
    }
    const F2 = this.computeForces();
    for (const [id, n] of this.nodes) {
      const a1 = scale(F.get(id)!, 1 / n.mass);
      const a2 = scale(F2.get(id)!, 1 / n.mass);
      n.vel = scale(add(n.vel, scale(add(a1, a2), 0.5 * this.dt)), this.damping);
    }
  }

  computeEnergiesAndAttention() {
    for (const n of this.nodes.values()) {
      let E = 0;
      for (const e of this.edges) {
        if (e.a === n.id || e.b === n.id) {
          const other = this.nodes.get(e.a === n.id ? e.b : e.a)!;
          const r = len(sub(n.pos, other.pos));
          const dl = r - e.L;
          E += 0.5 * e.k * dl * dl;
        }
      }
      n.energy = E;
      n.K *= this.knowledge_decay;
    }
    const raw = Array.from(this.nodes.values()).map(n => 6 * n.energy + 2 * n.K + gaussian() * 0.02);
    const maxv = Math.max(...raw);
    const exps = raw.map(v => Math.exp(v - maxv));
    const sum = exps.reduce((a, b) => a + b, 0) || 1;
    let idx = 0;
    for (const n of this.nodes.values()) {
      n.A = exps[idx++] / sum;
      const ell = this.replicate_lambda * ((n.K - this.theta_K) + (n.A - this.theta_A) - (n.energy - this.theta_E));
      n.repProb = 1 / (1 + Math.exp(-ell));
      if (n.repProb > Math.random()) this.replicate(n);
    }
  }

  semanticStep() {
    const newS = new Map<string, number[]>();
    for (const [id, n] of this.nodes) newS.set(id, n.sem.slice());
    for (const [id, n] of this.nodes) {
      const neigh = n.neighbors;
      const accum = new Array(this.D).fill(0);
      for (const j of neigh) {
        const sj = this.nodes.get(j)!.sem;
        for (let d = 0; d < this.D; d++) accum[d] += (sj[d] - n.sem[d]) / Math.max(1, neigh.length);
      }
      const out = newS.get(id)!;
      for (let d = 0; d < this.D; d++) out[d] = n.sem[d] + this.eta_sem * accum[d] + gaussian() * n.mutation_sigma;
    }
    for (const [id, arr] of newS) this.nodes.get(id)!.sem = arr;
  }

  replicate(n: NodeState) {
    const newPos = add(n.pos, scale([gaussian(), gaussian(), gaussian()], 0.5));
    const newSem = n.sem.map(v => v + gaussian() * 0.05);
    const newN = this.addNode(newPos, newSem);
    for (const neigh of n.neighbors.slice(0, Math.floor(n.neighbors.length / 2))) {
      this.addEdge(newN.id, neigh);
    }
  }

  step() {
    this.stepPhysics();
    this.computeEnergiesAndAttention();
    this.semanticStep();
  }

  snapshot() {
    return {
      nodes: Array.from(this.nodes.values()).map(n => ({
        id: n.id, pos: n.pos, vel: n.vel, E: n.energy, K: n.K, A: n.A, sem_head: n.sem.slice(0, 4), repProb: n.repProb
      })),
      edges: this.edges.map(e => [e.a, e.b, {k: e.k, L: e.L}]),
      stats: { totalEnergy: Array.from(this.nodes.values()).reduce((sum, n) => sum + n.energy, 0) }
    };
  }
