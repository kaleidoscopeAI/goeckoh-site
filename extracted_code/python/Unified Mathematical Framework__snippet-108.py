    from speech_loop import loop
    threading.Thread(target=loop.run, daemon=True).start()
    MainApp().run()

