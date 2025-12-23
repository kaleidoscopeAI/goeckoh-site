+ const meanK = node.K.reduce((a, b) => a + b, 0) / Math.max(1, node.K.length);
+ let var = 0;
+ for (let i = 0; i < node.K.length; i++) var += (node.K[i] - meanK) ** 2;
