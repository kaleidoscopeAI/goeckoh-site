"""
Return a parser for parsing requirement lines
"""
parser = optparse.OptionParser(add_help_option=False)

option_factories = SUPPORTED_OPTIONS + SUPPORTED_OPTIONS_REQ
for option_factory in option_factories:
    option = option_factory()
    parser.add_option(option)

# By default optparse sys.exits on parsing errors. We want to wrap
# that in our own exception.
def parser_exit(self: Any, msg: str) -> "NoReturn":
    raise OptionParsingError(msg)

# NOTE: mypy disallows assigning to a method
#       https://github.com/python/mypy/issues/2427
parser.exit = parser_exit  # type: ignore

return parser


