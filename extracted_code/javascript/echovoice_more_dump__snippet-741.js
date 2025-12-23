const snaps = await db.collection("projection_w_versions").orderBy("ts","desc").limit(50).get();
const fromDb = snaps.docs.map(d => d.data()).map((d:any) => ({ id: d.id, ts: d.ts?.seconds ? d.ts.seconds * 1000 : d.ts || 0, meta: d.meta
