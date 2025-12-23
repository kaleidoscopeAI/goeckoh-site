private _localSmooth(from: EVector, to: EVector) {
const duration = Math.max(10, this.smoothingMs);
const steps = Math.max(1, Math.floor(duration / 50));
let step = 0;
const timer = setInterval(() => {
