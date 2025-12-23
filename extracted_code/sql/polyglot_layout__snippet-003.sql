# compute median consensus φ over last second
phi_vals = [x[2] for x in window[-30:]] # recent lines
consensus = statistics.median(phi_vals)
ts = time.strftime("%Y-%m-%d %H:%M:%S")
con = sqlite3.connect(DB)
cur = con.cursor()
cur.execute("INSERT INTO consensus(ts, phi) VALUES (?, ?)", (ts, consensus))
con.commit(); con.close()
last_emit = now




