for (let i = 0; i < 10; i++) {
const r = doc(db, "actuation", `node_${i}_modulators`);
const unsub = onSnapshot(r, (snap) => {
