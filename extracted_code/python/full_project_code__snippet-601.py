"""
This metaclass automagically converts ``analyse_text`` methods into
static methods which always return float values.
"""

def __new__(mcs, name, bases, d):
    if 'analyse_text' in d:
        d['analyse_text'] = make_analysator(d['analyse_text'])
    return type.__new__(mcs, name, bases, d)


