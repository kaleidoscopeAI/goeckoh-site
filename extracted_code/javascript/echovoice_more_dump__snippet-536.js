private _notifySubscribers() {
const snapshot = copyVector(this.localE);
for (const cb of this.subscribers) cb(snapshot);
