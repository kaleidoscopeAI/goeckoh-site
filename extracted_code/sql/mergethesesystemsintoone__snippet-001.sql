con = sqlite3.connect(db_path); cur = con.cursor()
cur.execute("SELECT id, summary FROM facets")
out = {int(i): (s or "") for (i, s) in cur.fetchall()}
con.close(); return out

