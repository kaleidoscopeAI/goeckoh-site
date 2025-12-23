+ const id = `w_${Date.now()}`;
+ const payload = { id, ts: Date.now(), W, meta: meta || {} };
+ const file = path.join(W_DIR, `${id}.json`);
