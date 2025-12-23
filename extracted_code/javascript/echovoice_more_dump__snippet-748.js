+ for (let i = 0; i < node.K.length; i++) out.K[i] = lambdaPhi * (node.K[i] - mirrorTarget.K[i]);
+ for (let i = 0; i < node.D.length; i++) out.D[i] = lambdaPhi * (node.D[i] - mirrorTarget.D[i]);
