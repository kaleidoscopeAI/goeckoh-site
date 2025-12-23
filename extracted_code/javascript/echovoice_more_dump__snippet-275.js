let totalRegret = 0;
let emoRegretCorrelation = 0;
for (const t of logs) {
const regret = Math.max(0, t.optimalValue - t.actualValue);
