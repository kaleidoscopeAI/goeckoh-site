"""Will be raised if the backend is invalid."""
def __init__(self, backend_name, backend_path, message):
    super().__init__(message)
    self.backend_name = backend_name
    self.backend_path = backend_path


