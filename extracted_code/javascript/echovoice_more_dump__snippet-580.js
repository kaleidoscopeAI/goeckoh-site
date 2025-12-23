const sanitized: EVector = {} as any;
for (const emo of DEFAULT_EMOTIONS) sanitized[emo] = Math.max(-1, Math.min(1, e[emo] ?? 0));
const payload = { e: sanitized, ts: admin.firestore.FieldValue.serverTimestamp() };
