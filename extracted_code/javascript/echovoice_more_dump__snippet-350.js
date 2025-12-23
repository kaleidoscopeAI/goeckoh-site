const unsubNodes: (() => void)[] = [];
for (let i = 0; i < nodeCount; i++) {
const r = doc(db, "actuation", `node_${i}_modulators`);
const unsub = onSnapshot(r, (snap) => {
