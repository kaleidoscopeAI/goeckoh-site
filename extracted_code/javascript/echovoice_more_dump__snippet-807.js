+export function speciesFromE(W_A: number[][], E: EmotionalVector) {
+ const P = W_A.length;
+ const dE = E.values.length;
+ const s = new Array(P).fill(0);
+ for (let p = 0; p < P; p++) {
