const lrT = this.lr * Math.sqrt(1 - Math.pow(this.beta2, this.t)) / (1 - Math.pow(this.beta1, this.t));
for (const k in grad) {
for (let i = 0; i < grad[k].length; i++) {
