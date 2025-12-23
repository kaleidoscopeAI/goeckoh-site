class MemoryStore:
    def __init__(self, path: str, redis: aioredis.Redis):
        self.con = sqlite3.connect(path, check_same_thread=False)
        self.cur = self.con.cursor()
        self.cur.execute("CREATE TABLE IF NOT EXISTS dna (gen INTEGER PRIMARY KEY, dna TEXT)")
        self.cur.execute("CREATE INDEX IF NOT EXISTS idx_gen ON dna(gen)")
        self.cur.execute("CREATE TABLE IF NOT EXISTS insights (id TEXT PRIMARY KEY, data TEXT)")
        self.cur.execute("CREATE INDEX IF NOT EXISTS idx_id ON insights(id)")
        self.cur.execute("CREATE TABLE IF NOT EXISTS graph (source TEXT, target TEXT, weight REAL)")
        self.cur.execute("CREATE INDEX IF NOT EXISTS idx_source ON graph(source)")
        self.con.commit()
        self.redis = redis

    async def add_dna_batch(self, dnas: List[Tuple[int, KnowledgeDNA]]):
        data = [(gen, json.dumps(dna.__dict__, default=lambda o: str(o))) for gen, dna in dnas]
        async with self.redis.pipeline() as pipe:
            for gen, js in data:
                await pipe.set(f"agi:dna:{gen}", js, ex=CACHE_TTL)
            await pipe.execute()
        self.cur.executemany("INSERT OR REPLACE INTO dna VALUES (?, ?)", data)
        self.con.commit()

    async def get_dna(self, gen: int) -> KnowledgeDNA:
        cached = await self.redis.get(f"agi:dna:{gen}")
        if cached:
            data = json.loads(cached)
            return KnowledgeDNA(
                text_patterns=[PatternStrand(**p) for p in data['text_patterns']],
                visual_patterns=[VisualStrand(**v) for v in data['visual_patterns']],
                mutation_rate=data['mutation_rate'],
                generation=data['generation']
            )
        self.cur.execute("SELECT dna FROM dna WHERE gen=?", (gen,))
        row = self.cur.fetchone()
        if row:
            data = json.loads(row[0])
            await self.redis.set(f"agi:dna:{gen}", row[0], ex=CACHE_TTL)
            return KnowledgeDNA(
                text_patterns=[PatternStrand(**p) for p in data['text_patterns']],
                visual_patterns=[VisualStrand(**v) for v in data['visual_patterns']],
                mutation_rate=data['mutation_rate'],
                generation=data['generation']
            )
        return KnowledgeDNA()

    async def add_insight_batch(self, insights: List[Dict]):
        data = [(str(uuid.uuid4()), json.dumps(ins)) for ins in insights]
        async with self.redis.pipeline() as pipe:
            for id_, js in data:
                await pipe.set(f"agi:insight:{id_}", js, ex=CACHE_TTL)
            await pipe.execute()
        self.cur.executemany("INSERT INTO insights VALUES (?, ?)", data)
        self.con.commit()

    async def add_edge_batch(self, edges: List[Tuple[str, str, float]]):
        async with self.redis.pipeline() as pipe:
            for s, t, w in edges:
                await pipe.sadd(f"agi:graph:{s}", f"{t}:{w}")
            await pipe.execute()
        self.cur.executemany("INSERT INTO graph VALUES (?, ?, ?)", edges)
        self.con.commit()

