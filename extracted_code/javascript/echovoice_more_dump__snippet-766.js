let totalRegretEnd = 0;
for (let i = Math.max(0, logs.length - 10); i < logs.length; i++) totalRegretEnd += Math.abs(logs[i].regret);
const improvement = (totalRegretStart - totalRegretEnd) / (totalRegretStart || 1);
