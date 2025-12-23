"""
Return a `Formatter` subclass instance loaded from the provided file, relative
to the current directory.

The file is expected to contain a Formatter class named ``formattername``
(by default, CustomFormatter). Users should be very careful with the input, because
this method is equivalent to running ``eval()`` on the input file. The formatter is
given the `options` at its instantiation.

:exc:`pygments.util.ClassNotFound` is raised if there are any errors loading
the formatter.

.. versionadded:: 2.2
"""
try:
    # This empty dict will contain the namespace for the exec'd file
    custom_namespace = {}
    with open(filename, 'rb') as f:
        exec(f.read(), custom_namespace)
    # Retrieve the class `formattername` from that namespace
    if formattername not in custom_namespace:
        raise ClassNotFound('no valid %s class found in %s' %
                            (formattername, filename))
    formatter_class = custom_namespace[formattername]
    # And finally instantiate it with the options
    return formatter_class(**options)
except OSError as err:
    raise ClassNotFound('cannot read %s: %s' % (filename, err))
except ClassNotFound:
    raise
except Exception as err:
    raise ClassNotFound('error when loading custom formatter: %s' % err)


