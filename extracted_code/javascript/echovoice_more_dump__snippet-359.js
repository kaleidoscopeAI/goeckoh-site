const recent = this.reflectionLogs[nodeIdx].slice(-200);
const grad = computeMetaGrad(this.nodeStates[nodeIdx], recent);
