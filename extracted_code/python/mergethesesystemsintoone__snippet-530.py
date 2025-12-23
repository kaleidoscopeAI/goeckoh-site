def __init__(self, path: str):
    self.con = sqlite3.connect(path)
    self.cur = self.con.cursor()
    self.cur.execute("CREATE TABLE IF NOT EXISTS dna (gen INTEGER PRIMARY KEY, dna TEXT)")
    self.cur.execute("CREATE TABLE IF NOT EXISTS insights (id TEXT PRIMARY KEY, data TEXT)")
    self.cur.execute("CREATE TABLE IF NOT EXISTS graph (source TEXT, target TEXT, weight REAL)")

def add_dna(self, gen: int, dna: KnowledgeDNA):
    self.cur.execute("INSERT OR REPLACE INTO dna VALUES (?, ?)", (gen, json.dumps(dna.__dict__, default=lambda o: str(o))))
    self.con.commit()

def get_dna(self, gen: int) -> KnowledgeDNA:
    self.cur.execute("SELECT dna FROM dna WHERE gen=?", (gen,))
    row = self.cur.fetchone()
    if row:
        data = json.loads(row[0])
        return KnowledgeDNA(text_patterns=[PatternStrand(**p) for p in data['text_patterns']],
                            visual_patterns=[VisualStrand(**v) for v in data['visual_patterns']],
                            mutation_rate=data['mutation_rate'], generation=data['generation'])
    return KnowledgeDNA()

def add_insight(self, insight: Dict):
    id_ = str(uuid.uuid4())
    self.cur.execute("INSERT INTO insights VALUES (?, ?)", (id_, json.dumps(insight)))
    self.con.commit()

def add_edge(self, source: str, target: str, weight: float):
    self.cur.execute("INSERT INTO graph VALUES (?, ?, ?)", (source, target, weight))
    self.con.commit()

