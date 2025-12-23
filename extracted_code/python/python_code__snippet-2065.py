return Option(
    "-e",
    "--editable",
    dest="editables",
    action="append",
    default=[],
    metavar="path/url",
    help=(
        "Install a project in editable mode (i.e. setuptools "
        '"develop mode") from a local project path or a VCS url.'
    ),
)


