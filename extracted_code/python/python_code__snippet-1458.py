class Buffer(abc.ABC):
    """Base class for classes that implement the buffer protocol.

    The buffer protocol allows Python objects to expose a low-level
    memory buffer interface. Before Python 3.12, it is not possible
    to implement the buffer protocol in pure Python code, or even
    to check whether a class implements the buffer protocol. In
    Python 3.12 and higher, the ``__buffer__`` method allows access
    to the buffer protocol from Python code, and the
    ``collections.abc.Buffer`` ABC allows checking whether a class
    implements the buffer protocol.

    To indicate support for the buffer protocol in earlier versions,
    inherit from this ABC, either in a stub file or at runtime,
    or use ABC registration. This ABC provides no methods, because
    there is no Python-accessible methods shared by pre-3.12 buffer
    classes. It is useful primarily for static checks.

    """

# As a courtesy, register the most common stdlib buffer classes.
Buffer.register(memoryview)
Buffer.register(bytearray)
Buffer.register(bytes)


