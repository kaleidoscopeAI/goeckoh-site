let csv = "construct,activation,topNodeIds,topNodeContribs\n";
for (const s of summaries) {
const ids = s.topNodes.map((t:any)=>t.id).join("|");
const contribs = s.topNodes.map((t:any)=>t.contrib.toFixed(6)).join("|");
