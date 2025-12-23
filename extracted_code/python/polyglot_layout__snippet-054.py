      con.commit()
      con.close()

    def add_state(self, tick: int, tension: float, energy: float, size: int):
        con = sqlite3.connect(self.path)
        cur = con.cursor()
        cur.execute("INSERT INTO states(ts, tick, tension, energy, size) VALUES(?,?,?,?,?)", (time.time(), tick, tension, energy, size))
        con.commit()
        con.close()

    def add_reflection(self, tick: int, text: str):
        con = sqlite3.connect(self.path)
        cur = con.cursor()
        cur.execute("INSERT INTO reflections(ts, tick, text) VALUES(?,?,?)", (time.time(), tick, text))
        con.commit()
        con.close()

    def add_suggestion(self, tick: int, js: Dict[str, Any]):
        con = sqlite3.connect(self.path)
        cur = con.cursor()
        cur.execute("INSERT INTO suggestions(ts, tick, json) VALUES(?,?,?)", (time.time(), tick, json.dumps(js)))
        con.commit()
        con.close()

    def add_doc_with_embed(self, url: str, title: str, text: str):
        con = sqlite3.connect(self.path)
        cur = con.cursor()
        cur.execute("INSERT INTO docs(ts, url, title, text) VALUES(?,?,?,?)", (time.time(), url, title, text))
        doc_id = cur.lastrowid
        summary = (text.strip().split("\n")[0] if text else title)[:280]
        e = embed_text(text or title, D=self.D).astype(np.float32)
        cur.execute("INSERT OR REPLACE INTO facets(id, summary) VALUES(?,?)", (doc_id, summary))
        cur.execute("INSERT OR REPLACE INTO embeddings(id, vec) VALUES(?,?)", (doc_id, e.tobytes()))
        con.commit()
        con.close()
        return doc_id

    def add_energetics(self, tick: int, sigma: float, H_bits: float, S_field: float, L: float):
        con = sqlite3.connect(self.path)
        cur = con.cursor()
        cur.execute("INSERT INTO energetics(ts, tick, sigma, hbits, sfield, L) VALUES(?,?,?,?,?,?)", (time.time(), tick, sigma, H_bits, S_field,
