def python_implementation():
    """Return a string identifying the Python implementation."""
    if 'PyPy' in sys.version:
        return 'PyPy'
    if os.name == 'java':
        return 'Jython'
    if sys.version.startswith('IronPython'):
        return 'IronPython'
    return 'CPython'


