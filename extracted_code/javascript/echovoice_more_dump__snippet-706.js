let accum = zeros(d_N);
for (const nb of adj[i]) {
const j = nb.j;
const w = nb.w;
for (let d=0; d<d_N; d++) accum[d] += w * (V[j][d] - V[i][d]);
