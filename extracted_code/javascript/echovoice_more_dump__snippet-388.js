const maxC = Math.max(...node.chemicals);
const colorRatio = node.chemicals.map(c => c / (maxC || 1));
