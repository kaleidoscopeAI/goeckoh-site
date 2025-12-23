// 4. Compute gradient from recent reflections
const recent = this.reflectionLogs[nodeIdx].slice(-50); // last 50 reflections
const grad = computeMetaGrad(node, recent);
