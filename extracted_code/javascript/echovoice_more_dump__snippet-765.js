const batchSize = 1;
let totalRegretStart = 0;
for (let i = 0; i < Math.min(10, logs.length); i++) totalRegretStart += Math.abs(logs[i].regret);
for (let t = 0; t < logs.length; t++) {
