"""
Indicates a state combined from multiple states.
"""

def __new__(cls, *args):
    return tuple.__new__(cls, args)

def __init__(self, *args):
    # tuple.__init__ doesn't do anything
    pass


