const lrT = this.lr * Math.sqrt(1 - Math.pow(this.beta2, this.t)) / (1 - Math.pow(this.beta1, this.t));
for (let i = 0; i < N; i++) {
for (let j = 0; j < M; j++) {
const g = grad[i][j];
