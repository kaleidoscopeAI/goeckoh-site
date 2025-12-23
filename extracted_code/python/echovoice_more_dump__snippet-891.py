        # Fallback to mock purity function if C library is not available
        def compute_purity_mock(rho, N): return 0.5 + np.random.rand() * 0.5
        C_CORE = None
    else:
        C_CORE = ctypes.CDLL(LIB_PATH)
        # Define C function signatures for ctypes (essential for reliable calls)
        C_CORE.compute_purity.argtypes = [ctypes.c_void_p, ctypes.c_int]
        C_CORE.compute_purity.restype = ctypes.c_double
        print("C-Core (libkaleidoscope.so) loaded successfully.")

