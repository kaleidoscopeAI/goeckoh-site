class MemoryStore:
    def __init__(self, path: str, D: int = 512):
        self.path = path
        self.D = D
        self._init()

    def _init(self):
        con = sqlite3.connect(self.path)
        cur = con.cursor()
        # Existing tables
        # Add evolutionary
        cur.execute("""CREATE TABLE IF NOT EXISTS evo_states (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL, tick INTEGER, dna TEXT, phi REAL, generation INTEGER
        )""")
        con.commit()
        con.close()

    def add_evo_state(self, tick: int, dna: List[float], phi: float, generation: int):
        con = sqlite3.connect(self.path)
        cur = con.cursor()
        cur.execute("INSERT INTO evo_states(ts, tick, dna, phi, generation) VALUES(?,?,?,?,?)",
                    (time.time(), tick, json.dumps(dna), phi, generation))
        con.commit()
        con.close()

    # ... (other methods same)

