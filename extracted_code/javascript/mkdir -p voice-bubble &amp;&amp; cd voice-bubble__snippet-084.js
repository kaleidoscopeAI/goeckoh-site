const r0 = this._autocorrAtLag(x, 0) + 1e-12;
const rLag = this._autocorrAtLag(x, lag);

const ratio = clamp(rLag / r0, 1e-6, 0.999999);
const h = 10 * Math.log10(ratio / (1 - ratio));
return clamp(h, -20, 40);
}

