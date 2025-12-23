      con.commit(); con.close()
  def add_caption(self, tick: int, caption: str, top_ids: List[int], weights: List[float]):
      con = sqlite3.connect(self.path); cur = con.cursor()
      cur.execute("INSERT INTO captions(ts, tick, caption, top_ids, weights) VALUES(?,?,?,?,?)", (time.time(), tick, caption,
