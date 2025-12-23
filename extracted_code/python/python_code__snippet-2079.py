"""Custom help formatter for use in ConfigOptionParser.

This is updates the defaults before expanding them, allowing
them to show up correctly in the help listing.

Also redact auth from url type options
"""

def expand_default(self, option: optparse.Option) -> str:
    default_values = None
    if self.parser is not None:
        assert isinstance(self.parser, ConfigOptionParser)
        self.parser._update_defaults(self.parser.defaults)
        assert option.dest is not None
        default_values = self.parser.defaults.get(option.dest)
    help_text = super().expand_default(option)

    if default_values and option.metavar == "URL":
        if isinstance(default_values, str):
            default_values = [default_values]

        # If its not a list, we should abort and just return the help text
        if not isinstance(default_values, list):
            default_values = []

        for val in default_values:
            help_text = help_text.replace(val, redact_auth_from_url(val))

    return help_text


