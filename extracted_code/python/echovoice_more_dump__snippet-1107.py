# NOTE: You must compile kaleidoscope_core.c into a shared library first (e.g., libkaleidoscope.so)
# Compilation example: gcc -shared -o libkaleidoscope.so -fPIC kaleidoscope_core.c
LIB_PATH = os.path.join(os.path.dirname(__file__), "libkaleidoscope.so")
if not os.path.exists(LIB_PATH):
    print(f"WARNING: C library not found at {LIB_PATH}. Using mock function.")
    # Fallback to mock purity function if C library is not available
    def compute_purity_mock(rho, N): return 0.5 + np.random.rand() * 0.5
    C_CORE = None
else:
    C_CORE = ctypes.CDLL(LIB_PATH)
    # Define C function signatures for ctypes (essential for reliable calls)
    C_CORE.compute_purity.argtypes = [ctypes.c_void_p, ctypes.c_int]
    C_CORE.compute_purity.restype = ctypes.c_double
    print("C-Core (libkaleidoscope.so) loaded successfully.")

