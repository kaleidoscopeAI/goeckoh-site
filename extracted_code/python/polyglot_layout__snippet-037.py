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


