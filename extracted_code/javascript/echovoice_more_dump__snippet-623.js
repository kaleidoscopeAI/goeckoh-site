private notify() {
const snapshot = this.getConstructs();
for (const cb of this.subscribers) {
