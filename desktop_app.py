#!/usr/bin/env python3
"""
Goeckoh Desktop Application Entry Point
========================================

This is the main entry point for the Goeckoh desktop application.
It wires together all system components and launches the appropriate UI.
"""

import sys
import argparse
import logging
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.neuro_backend import NeuroKernel
import queue


def setup_logging(level="INFO"):
    """Configure application logging"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def launch_gui_mode(mode="child"):
    """Launch GUI application"""
    from main_app import ChildUI, run_clinician_dashboard
    
    logger = logging.getLogger(__name__)
    logger.info(f"Launching GUI in {mode} mode...")
    
    # Create UI queue for communication
    ui_queue = queue.Queue()
    
    # Initialize and start neuro backend
    logger.info("Initializing neuro backend...")
    kernel = NeuroKernel(ui_queue=ui_queue)
    kernel.start()
    logger.info("Neuro backend started")
    
    # Launch appropriate UI
    if mode == "child":
        logger.info("Starting child interface...")
        app = ChildUI(ui_queue)
        app.run()
    elif mode == "clinician":
        logger.info("Starting clinician dashboard...")
        run_clinician_dashboard(ui_queue)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(
        description="Goeckoh - Identity-Matched Speech Replay System"
    )
    parser.add_argument(
        "--mode",
        choices=["child", "clinician"],
        default="child",
        help="Interface mode (default: child)"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    logger.info("Starting Goeckoh Desktop Application...")
    logger.info(f"Mode: {args.mode}")
    
    try:
        # Launch the application
        launch_gui_mode(args.mode)
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
