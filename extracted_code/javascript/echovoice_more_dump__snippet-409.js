function generateConnections(nodeCount: number) {
const connections: [number, number][] = [];
for (let i = 0; i < nodeCount; i++) {
for (let j = i + 1; j < nodeCount; j++) {
