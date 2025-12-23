const dw = nb.computeDwAlongV(V[i], V[j], V, i, j); // returns scalar
for (let d=0; d<d_N; d++) Y[i][d] += η * dw * (N[j][d] - N[i][d]);
