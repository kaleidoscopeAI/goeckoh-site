const lrT = this.lr * Math.sqrt(1 - Math.pow(this.beta2, this.t)) / (1 - Math.pow(this.beta1, this.t));
for (const k of this.targets) {
const g = grad[k];
const m = this.m[k];
const v = this.v[k];
for (let i = 0; i < g.length; i++) {
