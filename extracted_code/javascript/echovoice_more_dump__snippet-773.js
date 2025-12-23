const maxDelta = Math.max(1e-9, ...constructDelta.map(Math.abs));
for (let j = 0; j < M; j++) constructDelta[j] /= maxDelta;
