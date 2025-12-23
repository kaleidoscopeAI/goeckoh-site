         insights.append(insight)

    await self.memory.add_insight_batch(insights)

    return insights

  async def _fetch_url(self, session: aiohttp.ClientSession, url: str) -> str:

    async with session.get(url, timeout=10) as resp:

         if resp.status == 200:

           html = await resp.text()

           return BeautifulSoup(html, 'html.parser').get_text()

         return ""

  async def introspect(self, insight: Dict):

    state = json.dumps({"phi": self.phi, "gen": self.dna.generation, "history_len": len(self.history)})

    prompts = [f"Introspect: {state} with {insight['content'][:100]}"]

    reflections = await llm_generate(prompts)

    print(f"Reflection: {reflections[0]}")

    await self.memory.add_edge_batch([(insight['id'], "self", 1.0)])

# Server

app = FastAPI(title="Groundbreaking AGI")

redis = None

@app.on_event("startup")

async def startup():

  global redis

  try:

    redis = await aioredis.create_redis_pool(REDIS_URL)

    await redis.ping()

  except:

    redis = None # Fallback to dict in MemoryStore

  await init_llm()

@app.on_event("shutdown")

async def shutdown():

  agi.pool.close()

  agi.pool.join()

  if redis:

    redis.close()

    await redis.wait_closed()

  loop = asyncio.get_running_loop()

  await loop.shutdown_asyncgens()


