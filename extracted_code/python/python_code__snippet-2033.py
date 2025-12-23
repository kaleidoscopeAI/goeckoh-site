"""Handle a single parsed requirements line; This can result in
creating/yielding requirements, or updating the finder.

:param line:        The parsed line to be processed.
:param options:     CLI options.
:param finder:      The finder - updated by non-requirement lines.
:param session:     The session - updated by non-requirement lines.

Returns a ParsedRequirement object if the line is a requirement line,
otherwise returns None.

For lines that contain requirements, the only options that have an effect
are from SUPPORTED_OPTIONS_REQ, and they are scoped to the
requirement. Other options from SUPPORTED_OPTIONS may be present, but are
ignored.

For lines that do not contain requirements, the only options that have an
effect are from SUPPORTED_OPTIONS. Options from SUPPORTED_OPTIONS_REQ may
be present, but are ignored. These lines may contain multiple options
(although our docs imply only one is supported), and all our parsed and
affect the finder.
"""

if line.is_requirement:
    parsed_req = handle_requirement_line(line, options)
    return parsed_req
else:
    handle_option_line(
        line.opts,
        line.filename,
        line.lineno,
        finder,
        options,
        session,
    )
    return None


