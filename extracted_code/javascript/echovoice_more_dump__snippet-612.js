let changed = false;
for (let i=0;i<W.length;i++) for (let j=0;j<W[0].length;j++) if (Math.abs(W[i][j] - before[i][j]) > 1e-8) changed = true;
