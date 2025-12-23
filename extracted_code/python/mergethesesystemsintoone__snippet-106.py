async def shutdown():
    redis.close()
    await redis.wait_closed()

