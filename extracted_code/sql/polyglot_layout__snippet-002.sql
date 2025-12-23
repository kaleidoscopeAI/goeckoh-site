     # persist snapshot
     con = sqlite3.connect(DB)
     cur = con.cursor()
     cur.execute("INSERT INTO snapshots(ts, source, phi, gen, conscious) VALUES (?, ?, ?, ?, ?)", (ts, name, phi, gen, conscious))
     con.commit(); con.close()

