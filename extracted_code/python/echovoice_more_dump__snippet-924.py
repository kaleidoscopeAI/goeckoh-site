async def start_crawl():
    # start crawler in background
    task = asyncio.create_task(crawler.start(interval=2.0))
    bg_tasks.append(task)
    return jsonify({'status': 'crawler_started'})

