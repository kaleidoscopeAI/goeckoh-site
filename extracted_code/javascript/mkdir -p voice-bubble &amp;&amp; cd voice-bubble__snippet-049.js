const r0 = autocorrAtLag(x, 0) + 1e-12;
const rLag = autocorrAtLag(x, lag);

const ratio = clamp(rLag / r0, 1e-6, 0.999999);
