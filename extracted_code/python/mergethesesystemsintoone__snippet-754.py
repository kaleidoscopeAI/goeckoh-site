# Load the shared library
c_lib = ctypes.CDLL("./c_backend/node_operations.so")

def __init__(self):
    self.nodes = {}
    self.failed_nodes = {}
    self._setup_c_functions()

def _setup_c_functions(self):
    """Defines argument types and return types for C functions."""
    self.c_lib.initialize_node.argtypes = [ctypes.c_char_p,




