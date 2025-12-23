+ const pending = { id, user, ts: Date.now() };
+ const file = path.join(W_DIR, `rollback_pending_${id}.json`);
