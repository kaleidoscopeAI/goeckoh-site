def __init__(self, path: str):
    self.con = sqlite3.connect(path)
    self.cur = self.con.cursor()
    self.cur.execute("CREATE TABLE IF NOT EXISTS dna (gen INTEGER PRIMARY KEY, dna TEXT)")
    self.cur.execute("CREATE TABLE IF NOT EXISTS insights (id INTEGER PRIMARY KEY, data TEXT)")

def add_dna(self, gen: int, dna: KnowledgeDNA):
    self.cur.execute("INSERT INTO dna VALUES (?, ?)", (gen, json.dumps(dna.__dict__)))
    self.con.commit()

def get_latest_dna(self) -> KnowledgeDNA:
    self.cur.execute("SELECT dna FROM dna ORDER BY gen DESC LIMIT 1")
    row = self.cur.fetchone()
    if row:
        return KnowledgeDNA(**json.loads(row[0]))
    return KnowledgeDNA()

