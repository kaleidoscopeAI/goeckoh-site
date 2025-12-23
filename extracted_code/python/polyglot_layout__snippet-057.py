      con = sqlite3.connect(self.path)
      cur = con.cursor()
      cur.execute("SELECT id, vec FROM embeddings ORDER BY id ASC")
      rows = cur.fetchall()
      con.close()
      if not rows:
          return np.zeros((0, 0), dtype=np.float64), []
      ids = [int(r[0]) for r in rows]
      arrs = [np.frombuffer(r[1], dtype=np.float32).astype(np.float64) for r in rows]
      E = np.stack(arrs, axis=0)
      E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
      if max_items and len(ids) > max_items:
          idx = np.random.RandomState(123).choice(len(ids), size=max_items, replace=False)
          E = E[idx]
          ids = [ids[i] for i in idx]
      return E, ids

  def recent(self, table: str, limit: int = 50):
      con = sqlite3.connect(self.path)
      cur = con.cursor()
      cur.execute(f"SELECT * FROM {table} ORDER BY id DESC LIMIT ?", (limit,))
      rows = cur.fetchall()
      con.close()
      return rows

