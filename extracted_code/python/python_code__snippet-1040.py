def exists_action() -> Option:
    return Option(
        # Option when path already exist
        "--exists-action",
        dest="exists_action",
        type="choice",
        choices=["s", "i", "w", "b", "a"],
        default=[],
        action="append",
        metavar="action",
        help="Default action when a path already exists: "
        "(s)witch, (i)gnore, (w)ipe, (b)ackup, (a)bort.",
    )


