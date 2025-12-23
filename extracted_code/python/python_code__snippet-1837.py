"""Get the representation of the specified wheel's distribution metadata.

This returns a Distribution instance from the chosen backend based on
the given wheel's ``.dist-info`` directory.

:param canonical_name: Normalized project name of the given wheel.
"""
return select_backend().Distribution.from_wheel(wheel, canonical_name)


