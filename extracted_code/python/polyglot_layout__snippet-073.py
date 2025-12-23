      con.commit(); con.close()
  def teach(self, text:str, lang:str):
      con=sqlite3.connect(self.path); cur=con.cursor()
      cur.execute("INSERT INTO facts(ts,lang,text) VALUES(?,?,?)",(time.time(), lang, text))
      fid=cur.lastrowid
      e=embed_text(text, D=self.D).astype(np.float32)
      cur.execute("INSERT OR REPLACE INTO embeds(id,vec) VALUES(?,?)",(fid, e.tobytes()))
      con.commit(); con.close(); return fid
  def embeddings(self, max_items:Optional[int]=None)->Tuple[np.ndarray,List[int]]:
      con=sqlite3.connect(self.path); cur=con.cursor()
      cur.execute("SELECT id,vec FROM embeds ORDER BY id ASC"); rows=cur.fetchall(); con.close()
      if not rows: return np.zeros((0,0),dtype=np.float64), []
      ids=[int(r[0]) for r in rows]; arr=[np.frombuffer(r[1],dtype=np.float32).astype(np.float64) for r in rows]
      E=np.stack(arr,axis=0); E/= (np.linalg.norm(E,axis=1,keepdims=True)+1e-9)
      if max_items and len(ids)>max_items:
          idx=np.random.RandomState(123).choice(len(ids), size=max_items, replace=False)
          E=E[idx]; ids=[ids[i] for i in idx]
      return E, ids
  def log(self, tick:int, type_:str, data:dict):
      con=sqlite3.connect(self.path); cur=con.cursor()
      cur.execute("INSERT INTO traces(ts,tick,type,json) VALUES(?,?,?,?)",(time.time(), tick, type_, json.dumps(data)))
      con.commit(); con.close()

