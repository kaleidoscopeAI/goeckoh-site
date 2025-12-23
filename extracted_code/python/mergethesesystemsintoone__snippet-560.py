def __init__(self, path: str, redis: Optional[aioredis.Redis]):
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
    self.fallback_cache = {}  # Dict fallback

async def _get_cache(self, key: str):
    if self.redis:
        try:
            return await self.redis.get(key)
        except:
            pass
    return self.fallback_cache.get(key)

async def _set_cache(self, key: str, value: str, ex: int):
    if self.redis:
        try:
            await self.redis.set(key, value, ex=ex)
            return
        except:
            pass
    self.fallback_cache[key] = value

async def add_dna_batch(self, dnas: List[Tuple[int, KnowledgeDNA]]):
    data = [(gen, json.dumps(dna.__dict__, default=lambda o: str(o))) for gen, dna in dnas]
    if self.redis:
        async with self.redis.pipeline() as pipe:
            for gen, js in data:
                await pipe.set(f"agi:dna:{gen}", js, ex=CACHE_TTL)
            await pipe.execute()
    else:
        for gen, js in data:
            self.fallback_cache[f"agi:dna:{gen}"] = js
    self.cur.executemany("INSERT OR REPLACE INTO dna VALUES (?, ?)", data)
    self.con.commit()

async def get_dna(self, gen: int) -> KnowledgeDNA:
    cached = await self._get_cache(f"agi:dna:{gen}")
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
        await self._set_cache(f"agi:dna:{gen}", row[0], CACHE_TTL)
        return KnowledgeDNA(
            text_patterns=[PatternStrand(**p) for p in data['text_patterns']],
            visual_patterns=[VisualStrand(**v) for v in data['visual_patterns']],
            mutation_rate=data['mutation_rate'],
            generation=data['generation']
        )
    return KnowledgeDNA()

async def add_insight_batch(self, insights: List[Dict]):
    data = [(str(uuid.uuid4()), json.dumps(ins)) for ins in insights]
    if self.redis:
        async with self.redis.pipeline() as pipe:
            for id_, js in data:
                await pipe.set(f"agi:insight:{id_}", js, ex=CACHE_TTL)
            await pipe.execute()
    else:
        for id_, js in data:
            self.fallback_cache[f"agi:insight:{id_}"] = js
    self.cur.executemany("INSERT INTO insights VALUES (?, ?)", data)
    self.con.commit()

async def add_edge_batch(self, edges: List[Tuple[str, str, float]]):
    if self.redis:
        async with self.redis.pipeline() as pipe:
            for s, t, w in edges:
                await pipe.sadd(f"agi:graph:{s}", f"{t}:{w}")
            await pipe.execute()
    else:
        for s, t, w in edges:
            if f"agi:graph:{s}" not in self.fallback_cache:
                self.fallback_cache[f"agi:graph:{s}"] = set()
            self.fallback_cache[f"agi:graph:{s}"].add(f"{t}:{w}")
    self.cur.executemany("INSERT INTO graph VALUES (?, ?, ?)", edges)
    self.con.commit()

