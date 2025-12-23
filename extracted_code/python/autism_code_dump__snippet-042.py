def main():
    start_api_server()
    try:
        run_echo_loop()
    except KeyboardInterrupt:
        print("\n[Main] KeyboardInterrupt – stopping.")
    finally:
        global RUNNING
        RUNNING = False


