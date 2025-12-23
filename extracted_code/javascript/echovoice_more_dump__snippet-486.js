const maxC = Math.max(...node.chemicals);
const ratios = node.chemicals.map(c => c / (maxC || 1));
