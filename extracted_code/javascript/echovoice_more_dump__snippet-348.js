const nodeIdx = this.reflectionQueue.shift()!;
const logs = this.reflectionLogs[nodeIdx] || [];
const recent = logs.slice(-200);
