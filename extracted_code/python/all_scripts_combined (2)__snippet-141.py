# package, so we import from sibling modules using a leading dot.
from .config import CONFIG, CompanionConfig
from .speech_loop import SpeechLoop
from .tk_gui import run_gui


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Jackson's Companion command-line interface"
    )
    sub = parser.add_subparsers(dest="command")
    # run subcommand: run the speech loop in CLI
    sub.add_parser("run", help="Run the speech loop (simulation) in the console")
    # gui subcommand: launch the GUI
    sub.add_parser(
        "gui", help="Launch the desktop GUI (parent + child views)"
    )
    args = parser.parse_args(argv)
    if args.command == "run":
        loop = SpeechLoop(CONFIG)
        # Run the asynchronous loop in a blocking fashion
        try:
            asyncio.run(loop.run())
        except KeyboardInterrupt:
            print("Speech loop stopped.")
    elif args.command == "gui":
        run_gui(CONFIG)
    else:
        parser.print_help()


