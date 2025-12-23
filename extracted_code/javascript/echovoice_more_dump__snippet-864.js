+ const auditFile = path.join(W_DIR, `audit_log.jsonl`);
+ const record = { id, approver, ts: Date.now() };
