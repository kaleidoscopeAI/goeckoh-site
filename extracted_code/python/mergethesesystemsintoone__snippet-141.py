async def shutdown():
    agi.pool.close()
    agi.pool.join()
    if redis:
        redis.close()
        await redis.wait_closed()
    loop = asyncio.get_running_loop()
    await loop.shutdown_asyncgens()

