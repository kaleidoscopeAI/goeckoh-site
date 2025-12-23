async def startup():
    global redis
    try:
        redis = await aioredis.create_redis_pool(REDIS_URL)
        await redis.ping()
    except:
        redis = None  # Fallback to dict in MemoryStore
    await init_llm()

