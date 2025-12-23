const norm = Math.sqrt(v.reduce((acc, x) => acc + x * x, 0));
if (norm > 0) for (let i = 0; i < dim; i++) v[i] /= norm;
