const species: number[] = Array(numSpecies).fill(0);
for (let i = 0; i < numSpecies; i++) {
let sum = 0;
for (let j = 0; j < numEmotions; j++) sum += (e[DEFAULT_EMOTIONS[j]] ?? 0) * W_actuation[i][j];
