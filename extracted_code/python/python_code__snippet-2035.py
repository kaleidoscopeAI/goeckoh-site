def parse_line(line: str) -> Tuple[str, Values]:
    # Build new parser for each line since it accumulates appendable
    # options.
    parser = build_parser()
    defaults = parser.get_default_values()
    defaults.index_url = None
    if finder:
        defaults.format_control = finder.format_control

    args_str, options_str = break_args_options(line)

    try:
        options = shlex.split(options_str)
    except ValueError as e:
        raise OptionParsingError(f"Could not split options: {options_str}") from e

    opts, _ = parser.parse_args(options, defaults)

    return args_str, opts

return parse_line


