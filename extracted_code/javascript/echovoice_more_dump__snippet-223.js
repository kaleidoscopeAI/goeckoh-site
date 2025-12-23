const Lv = SparseMatrix.mul(this.graph.L, v); // returns Float32Array[n]
for (let i = 0; i < n; i++) out[i * m + k] += -Lv[i];
