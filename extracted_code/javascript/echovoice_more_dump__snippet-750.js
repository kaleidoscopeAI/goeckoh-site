+ const L = node.L;
+ const s = Math.tanh(L.reduce((a, b) => a + b, 0) / Math.max(1, L.length));
+ for (let i = 0; i < out.K.length; i++) out.K[i] = gammaSpec * s * (1 - Math.exp(-Math.abs(node.K[i])));
+ for (let i = 0; i < out.D.length; i++) out.D[i] = gammaSpec * s * (1 - Math.exp(-Math.abs(node.D[i])));
