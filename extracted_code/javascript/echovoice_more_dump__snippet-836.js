function stress_Jv(v_flat) {
const V = view2D(v_flat, m, d_N);
const Y = zeros2D(m, d_N);
for (let i=0;i<m;i++){
for (const nb of adj[i]) {
const j = nb.j;
