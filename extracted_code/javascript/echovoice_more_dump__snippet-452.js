const recent = this.reflectionLogs[nodeIdx].slice(-50);
const grad = computeMetaGrad(node, recent);
