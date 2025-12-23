def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Real Unified Neuro-Acoustic AGI System")
    parser.add_argument("--config", default="real_system_config.ini", help="Path to config file")
    parser.add_argument("--api", action="store_true", help="Run without CLI prompt (API/service mode)")
    parser.add_argument("--no-interactive", action="store_true", help="Alias for --api; skip CLI loop")
    parser.add_argument("--disable-api", action="store_true", help="Do not start the Flask API server")
    args = parser.parse_args()

    print("🚀 Starting Real Unified Neuro-Acoustic AGI System...")

    # Auto-disable interactive loop if stdin is not a TTY (e.g., CI/headless exec)
    if not sys.stdin.isatty() and not (args.api or args.no_interactive):
        print("ℹ️  No TTY detected; running in API mode (--api)")
        args.api = True
    
    # Initialize system
    system = RealUnifiedSystem(config_file=args.config, disable_api=args.disable_api)
    
    # Start system
    system.start()
    
    try:
        if args.api or args.no_interactive:
            print("🌐 API mode: running without interactive prompt")
            while True:
                time.sleep(1.0)
        else:
            # Run interactive mode
            system.run_interactive_mode()
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    finally:
        # Stop system
        system.stop()
        print("👋 System shutdown complete")

