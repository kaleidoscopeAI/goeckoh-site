    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=int, default=100)
    args = parser.parse_args()
    system = Kaleidoscope()
    system.start(cycles=args.cycles)




