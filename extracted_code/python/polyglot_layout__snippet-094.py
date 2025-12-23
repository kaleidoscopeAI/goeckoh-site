  def __init__(self, path: str, D: int = 512):
      self.path = path; self.D = D; self._init()
  def _init(self):
      con = sqlite3.connect(self.path); cur = con.cursor()
      cur.execute("""CREATE TABLE IF NOT EXISTS states (id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, tick INTEGER, tension REAL, energy
