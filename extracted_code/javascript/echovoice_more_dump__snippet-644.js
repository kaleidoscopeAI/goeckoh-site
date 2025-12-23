const norm = 1 / (1 + Math.exp(-a)); // logistic to compress extremes
const r = Math.round(Math.min(255, 255 * Math.pow(norm, 1.2)));
const g = Math.round(Math.min(255, 255 * (1 - Math.abs(norm - 0.5) * 2)));
const b = Math.round(Math.min(255, 255 * (1 - norm)));
