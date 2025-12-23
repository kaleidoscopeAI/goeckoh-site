if (!this.bitProbabilistic) {
  let diff = 0;
  for (let k = 0; k < this.dBit; k++) diff += ei[k] !== ej[k] ? 1 : 0;
  return 1 - diff / this.dBit;
} else {
  let sum = 0;
  for (let k = 0; k < this.dBit; k++) {
    sum += ei[k] * ej[k] * (2 * (ei[k] === ej[k] ? 1 : 0) - 1); // Approx delta
  }
  return sum / this.dBit;
}
