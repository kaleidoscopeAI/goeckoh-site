    import uvicorn
    import threading

    async def autonomous_loop():
        while True:
            await organic_ai.run_organic_cycle()
            await asyncio.sleep(0.5)

    # Optionally start autonomous loop in background when running directly
    # threading.Thread(target=lambda: asyncio.run(autonomous_loop()), daemon=True).start()

    uvicorn.run(app, host="127.0.0.1", port=5000)

