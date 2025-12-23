+async function main() {
+ const m = 64;
+ const nodes: NodeState[] = [];
+ for (let i = 0; i < m; i++) nodes.push(makeRandomNode(i));
