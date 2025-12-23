"""Load a lexer from a file.

This method expects a file located relative to the current working
directory, which contains a Lexer class. By default, it expects the
Lexer to be name CustomLexer; you can specify your own class name
as the second argument to this function.

Users should be very careful with the input, because this method
is equivalent to running eval on the input file.

Raises ClassNotFound if there are any problems importing the Lexer.

.. versionadded:: 2.2
"""
try:
    # This empty dict will contain the namespace for the exec'd file
    custom_namespace = {}
    with open(filename, 'rb') as f:
        exec(f.read(), custom_namespace)
    # Retrieve the class `lexername` from that namespace
    if lexername not in custom_namespace:
        raise ClassNotFound('no valid %s class found in %s' %
                            (lexername, filename))
    lexer_class = custom_namespace[lexername]
    # And finally instantiate it with the options
    return lexer_class(**options)
except OSError as err:
    raise ClassNotFound('cannot read %s: %s' % (filename, err))
except ClassNotFound:
    raise
except Exception as err:
    raise ClassNotFound('error when loading custom lexer: %s' % err)


