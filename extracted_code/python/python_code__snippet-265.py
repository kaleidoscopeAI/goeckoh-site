    from urllib.request import urlopen


def assert_lower(string):
    assert string == string.lower()
    return string


def generate(url):
    parts = ['''\
