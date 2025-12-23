function makeDeterministicW(nodeCount: number, constructs: string[], seed = 1337): number[][] {
const M = constructs.length;
const rnd = mulberry32(seed);
