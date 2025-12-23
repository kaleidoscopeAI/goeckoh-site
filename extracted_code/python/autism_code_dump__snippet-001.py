from __future__ import annotations

import argparse
import asyncio
from typing import Optional

from .config import CONFIG
from .speech_loop import SpeechLoop
from .tk_gui import run_gui


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="KQBC Agent command-line interface"
    )
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("run", help="Run the speech loop (simulation) in the console")
    sub.add_parser("gui", help="Launch the desktop GUI (parent + child views)")
    args = parser.parse_args(argv)
    if args.command == "run":
        loop = SpeechLoop(CONFIG)
        try:
            asyncio.run(loop.run())
        except KeyboardInterrupt:
            print("Speech loop stopped.")
    elif args.command == "gui":
        run_gui(CONFIG)
    else:
        parser.print_help()


