"""Get the distribution metadata representation in the specified directory.

This returns a Distribution instance from the chosen backend based on
the given on-disk ``.dist-info`` directory.
"""
return select_backend().Distribution.from_directory(directory)


