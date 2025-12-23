function autocorrAtLag(x, lag) {
let s = 0;
for (let i = 0; i < x.length - lag; i++) s += x[i] * x[i + lag];
