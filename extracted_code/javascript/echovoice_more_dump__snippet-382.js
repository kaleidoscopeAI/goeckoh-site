function projectVec(species: FloatArr, proj: number[]): number {
let out = 0;
for (let i = 0; i < species.length; i++) out += species[i] * proj[i];
