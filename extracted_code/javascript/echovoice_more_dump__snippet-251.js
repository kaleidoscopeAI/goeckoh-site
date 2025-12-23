// Project species using a dense vector
function projectVec(species: FloatArr, proj: number[]): number {
const m = species.length;
let out = 0;
for (let i = 0; i < m; i++) out += species[i] * proj[i];
