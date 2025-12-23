const [i, j] = connections[fIdx];
const srcNode = loop.nodeStates[i];
const dstNode = loop.nodeStates[j];
const intensity = Math.min(1, (Math.max(...srcNode.chemicals) + Math.max(...dstNode.chemicals)) / 2);
