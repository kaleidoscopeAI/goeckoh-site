def __init__(
    self,
    session: PipSession,
    line_parser: LineParser,
) -> None:
    self._session = session
    self._line_parser = line_parser

def parse(
    self, filename: str, constraint: bool
) -> Generator[ParsedLine, None, None]:
    """Parse a given file, yielding parsed lines."""
    yield from self._parse_and_recurse(filename, constraint)

def _parse_and_recurse(
    self, filename: str, constraint: bool
) -> Generator[ParsedLine, None, None]:
    for line in self._parse_file(filename, constraint):
        if not line.is_requirement and (
            line.opts.requirements or line.opts.constraints
        ):
            # parse a nested requirements file
            if line.opts.requirements:
                req_path = line.opts.requirements[0]
                nested_constraint = False
            else:
                req_path = line.opts.constraints[0]
                nested_constraint = True

            # original file is over http
            if SCHEME_RE.search(filename):
                # do a url join so relative paths work
                req_path = urllib.parse.urljoin(filename, req_path)
            # original file and nested file are paths
            elif not SCHEME_RE.search(req_path):
                # do a join so relative paths work
                req_path = os.path.join(
                    os.path.dirname(filename),
                    req_path,
                )

            yield from self._parse_and_recurse(req_path, nested_constraint)
        else:
            yield line

def _parse_file(
    self, filename: str, constraint: bool
) -> Generator[ParsedLine, None, None]:
    _, content = get_file_content(filename, self._session)

    lines_enum = preprocess(content)

    for line_number, line in lines_enum:
        try:
            args_str, opts = self._line_parser(line)
        except OptionParsingError as e:
            # add offending line
            msg = f"Invalid requirement: {line}\n{e.msg}"
            raise RequirementsFileParseError(msg)

        yield ParsedLine(
            filename,
            line_number,
            args_str,
            opts,
            constraint,
        )


