def signal_handler(signum, frame):
    global running
    print(f"\nInterrupt signal ({signum}) received. Shutting down gracefully...")
    running = False

