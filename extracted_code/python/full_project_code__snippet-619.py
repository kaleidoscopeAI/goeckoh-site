"""
Abstract class used by `simplefilter` to create simple
function filters on the fly. The `simplefilter` decorator
automatically creates subclasses of this class for
functions passed to it.
"""
function = None

def __init__(self, **options):
    if not hasattr(self, 'function'):
        raise TypeError('%r used without bound function' %
                        self.__class__.__name__)
    Filter.__init__(self, **options)

def filter(self, lexer, stream):
    # pylint: disable=not-callable
    yield from self.function(lexer, stream, self.options)


