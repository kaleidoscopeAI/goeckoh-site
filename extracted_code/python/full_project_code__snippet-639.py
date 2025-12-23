"""Highlight special code tags in comments and docstrings.

Options accepted:

`codetags` : list of strings
   A list of strings that are flagged as code tags.  The default is to
   highlight ``XXX``, ``TODO``, ``FIXME``, ``BUG`` and ``NOTE``.

.. versionchanged:: 2.13
   Now recognizes ``FIXME`` by default.
"""

def __init__(self, **options):
    Filter.__init__(self, **options)
    tags = get_list_opt(options, 'codetags',
                        ['XXX', 'TODO', 'FIXME', 'BUG', 'NOTE'])
    self.tag_re = re.compile(r'\b(%s)\b' % '|'.join([
        re.escape(tag) for tag in tags if tag
    ]))

def filter(self, lexer, stream):
    regex = self.tag_re
    for ttype, value in stream:
        if ttype in String.Doc or \
           ttype in Comment and \
           ttype not in Comment.Preproc:
            yield from _replace_special(ttype, value, regex, Comment.Special)
        else:
            yield ttype, value


