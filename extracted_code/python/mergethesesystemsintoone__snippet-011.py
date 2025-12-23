class Memory:
    def __init__(self, path:str, D:int=512):
        self.path=path; self.D=D; self._init()
    def _init(self):
        con=sqlite3.connect(self.path); cur=con.cursor()
        cur.execute("""CREATE TABLE IF NOT EXISTS facts(id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, lang TEXT, text TEXT)""")
        cur.execute("""CREATE TABLE IF NOT EXISTS embeds(id INTEGER PRIMARY KEY, vec BLOB)""")
        cur.execute("""CREATE TABLE IF NOT EXISTS traces(id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, tick INTEGER, type TEXT, json TEXT)""")
        cur.execute("""CREATE TABLE IF NOT EXISTS energetics(id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, tick INTEGER, sigma REAL, hbits 
