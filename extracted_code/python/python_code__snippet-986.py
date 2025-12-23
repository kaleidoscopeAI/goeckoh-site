def preprocess(content: str) -> ReqFileLines:
    """Split, filter, and join lines, and return a line iterator

    :param content: the content of the requirements file
    """
    lines_enum: ReqFileLines = enumerate(content.splitlines(), start=1)
    lines_enum = join_lines(lines_enum)
    lines_enum = ignore_comments(lines_enum)
    lines_enum = expand_env_variables(lines_enum)
    return lines_enum


def handle_requirement_line(
    line: ParsedLine,
    options: Optional[optparse.Values] = None,
