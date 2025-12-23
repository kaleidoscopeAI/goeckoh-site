def main():
    parser = argparse.ArgumentParser(description="Kaleidoscope AI Controller")
    parser.add_argument("--setup", action="store_true", help="Setup environment only")
    parser.add_argument("--start", action="store_true", help="Start all components")
    parser.add_argument("--stop", action="store_true", help="Stop all components")
    
    args = parser.parse_args()
    
    # Setup environment if requested or if starting components
    if args.setup or args.start:
        if not setup_environment():
            logger.error("Environment setup failed")
            return 1
    
    # Start or stop components
    if args.start:
        config_path = os.environ.get("KALEIDOSCOPE_CONFIG", "config.json")
        manager = ComponentManager(config_path)
        
        # Start task manager first
        manager.start_component("task_manager", "src/utils/task_manager.py")
        time.sleep(2)  # Give task manager time to initialize
        
        # Start LLM service
        manager.start_component("llm_service", "src/core/llm_service.py")
        time.sleep(2)  # Give LLM service time to initialize
        
        # Start API server
        manager.start_api_server()
        
        logger.info("All components started")
        
        try:
            # Keep running until interrupted
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            manager.stop_all()
    
    elif args.stop:
        config_path = os.environ.get("KALEIDOSCOPE_CONFIG", "config.json")
        manager = ComponentManager(config_path)
        manager.stop_all()
        logger.info("All components stopped")
    
    return 0

