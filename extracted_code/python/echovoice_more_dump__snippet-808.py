def load_rust(lib_path=None):
    global _lib
    lib_path = lib_path or os.environ.get("RUST_KERNEL_PATH", "rust_kernel/target/release/librust_kernel.so")
    if not os.path.exists(lib_path):
        raise FileNotFoundError(lib_path)
    _lib = ctypes.CDLL(lib_path)
    _lib.eval_energy.restype = ctypes.c_double
    _lib.eval_energy.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int]
    _lib.parallel_tempering.restype = ctypes.c_int
    _lib.parallel_tempering.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int]
    return _lib

def eval_energy(vec: np.ndarray):
    global _lib
    if _lib is None:
        load_rust()
    arr = np.ascontiguousarray(vec.astype(np.float64))
    ptr = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    return float(_lib.eval_energy(ptr, arr.size))

def parallel_tempering(states: np.ndarray):
    global _lib
    if _lib is None:
        load_rust()
    arr = np.ascontiguousarray(states.astype(np.float64).ravel())
    ptr = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    n_states, dim = states.shape
    return int(_lib.parallel_tempering(ptr, n_states, dim))

