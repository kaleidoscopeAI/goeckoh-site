    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Pretty print json")
    parser.add_argument(
        "path",
        metavar="PATH",
        help="path to file, or - for stdin",
    )
    parser.add_argument(
        "-i",
        "--indent",
        metavar="SPACES",
        type=int,
        help="Number of spaces in an indent",
        default=2,
    )
    args = parser.parse_args()

    from pip._vendor.rich.console import Console

    console = Console()
    error_console = Console(stderr=True)

    try:
        if args.path == "-":
            json_data = sys.stdin.read()
        else:
            json_data = Path(args.path).read_text()
    except Exception as error:
        error_console.print(f"Unable to read {args.path!r}; {error}")
        sys.exit(-1)

    console.print(JSON(json_data, indent=args.indent), soft_wrap=True)


