    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=5001)
    parser.add_argument('--autonomous', action='store_true')
    args = parser.parse_args()

    # ensure corpus exists and embedder is fitted
    ensure_corpus('./corpus')
    # fit embedder on corpus
    texts = []
    for f in os.listdir('./corpus'):
        if f.endswith('.txt'):
            with open(os.path.join('./corpus', f), 'r', encoding='utf-8') as fh:
                texts.append(fh.read())
    if texts:
        core.embedder.fit(texts)

    loop = asyncio.get_event_loop()
    if args.autonomous:
        loop.create_task(autonomous_loop())
    # start Quart
    app.run(host=args.host, port=args.port)

