const choices = [
  `Quantum coherence spike in node ${triggerNode}`,
  `Phase synchronization near sector ${triggerNode % 8}`
];
const h = { text: choices[Math.floor(Math.random() * choices.length)], timestamp: Date.now(), confidence: this.globalCoherence, triggerNode };
this.hypothesisHistory.push(h);
return h;
