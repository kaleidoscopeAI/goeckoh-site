const species: number[] = Array(numSpecies).fill(0);
for (let i = 0; i < numSpecies; i++) {
let sum = 0;
for (let j = 0; j < numEmotions; j++) {
const emotion = DEFAULT_EMOTIONS[j];
