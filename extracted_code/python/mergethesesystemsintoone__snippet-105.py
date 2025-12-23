async def startup():
    global redis
    redis = await aioredis.create_redis_pool(REDIS_URL)

