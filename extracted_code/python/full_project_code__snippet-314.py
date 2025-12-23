def osname(name):
    orig = os.name
    os.name = name
    yield
    os.name = orig

