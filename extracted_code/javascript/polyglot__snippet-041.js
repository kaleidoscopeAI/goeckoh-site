const store = new MemoryStore(db);
store.insertStmt = await db.prepare('INSERT OR REPLACE INTO dna (gen, phi) VALUES (?, ?)');
