def __init__(self):
    # the subclasses declare class attributes which are numbers.
    # Upon instantiation we define instance attributes, which are the same
    # as the class attributes but wrapped with the ANSI escape sequence
    for name in dir(self):
        if not name.startswith('_'):
            value = getattr(self, name)
            setattr(self, name, code_to_chars(value))


