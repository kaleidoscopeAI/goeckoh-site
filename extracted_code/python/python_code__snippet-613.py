class EmptyProvider(NullProvider):
    """Provider that returns nothing for all requests"""

    module_path = None

    _isdir = _has = lambda self, path: False

    def _get(self, path):
        return ''

    def _listdir(self, path):
        return []

    def __init__(self):
        pass


