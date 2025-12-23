"""Convert keywords to lowercase or uppercase or capitalize them, which
means first letter uppercase, rest lowercase.

This can be useful e.g. if you highlight Pascal code and want to adapt the
code to your styleguide.

Options accepted:

`case` : string
   The casing to convert keywords to. Must be one of ``'lower'``,
   ``'upper'`` or ``'capitalize'``.  The default is ``'lower'``.
"""

def __init__(self, **options):
    Filter.__init__(self, **options)
    case = get_choice_opt(options, 'case',
                          ['lower', 'upper', 'capitalize'], 'lower')
    self.convert = getattr(str, case)

def filter(self, lexer, stream):
    for ttype, value in stream:
        if ttype in Keyword:
            yield ttype, self.convert(value)
        else:
            yield ttype, value


