return Option(
    "-r",
    "--requirement",
    dest="requirements",
    action="append",
    default=[],
    metavar="file",
    help="Install from the given requirements file. "
    "This option can be used multiple times.",
)


