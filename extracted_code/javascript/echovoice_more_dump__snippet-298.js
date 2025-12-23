const n = this.graph.n, m = this.cfg.speciesCount;
const N = n * m;
const k1 = new Float32Array(N), k2 = new Float32Array(N), k3 = new Float32Array(N), k4 = new Float32Array(N);
const backup = new Float32Array(this.S);
