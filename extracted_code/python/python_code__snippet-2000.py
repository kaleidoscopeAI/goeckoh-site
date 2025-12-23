"""
Show information about one or more installed packages.

The output is in RFC-compliant mail header format.
"""

usage = """
  %prog [options] <package> ..."""
ignore_require_venv = True

def add_options(self) -> None:
    self.cmd_opts.add_option(
        "-f",
        "--files",
        dest="files",
        action="store_true",
        default=False,
        help="Show the full list of installed files for each package.",
    )

    self.parser.insert_option_group(0, self.cmd_opts)

def run(self, options: Values, args: List[str]) -> int:
    if not args:
        logger.warning("ERROR: Please provide a package name or names.")
        return ERROR
    query = args

    results = search_packages_info(query)
    if not print_results(
        results, list_files=options.files, verbose=options.verbose
    ):
        return ERROR
    return SUCCESS


