async def speculate_step(snapshot: Snapshot):
    """
    Core cognitive step: Ingests sensor data, calculates geometric flow,
    and updates system state based on the Forman-Ricci Flow (FRF) analog.
    """
    
    # 1. Process Sensor Data (Speculation Engine)
    kappa = snapshot.system_metrics.get("kappa", 1.0)
    sensor_update = sensor_grad(kappa, snapshot.adc_raw).tolist()

    # 2. Geometric Core (C-Backend Call)
    # Mock data for C-call (Density Matrix for Purity calculation)
    N = snapshot.system_metrics.get("node_count", SYSTEM_STATE["node_count"])
    # Create a mock flattened density matrix (N*N complex values)
    mock_rho_data = [complex(np.random.rand(), np.random.rand()) for _ in range(N*N)] 

    if C_CORE:
        # Prepare C types for complex array
        class ComplexArray_c(ctypes.Structure):
            _fields_ = [("n", ctypes.c_int), ("data", ctypes.POINTER(ctypes.c_double_complex))]

        rho_ptr = (ctypes.c_double_complex * len(mock_rho_data))(*mock_rho_data)
        
        # NOTE: C-side allocation would be preferred, but for simple mock passing:
        class ComplexArray_c_ptr(ctypes.Structure):
            _fields_ = [("n", ctypes.c_int), ("data", ctypes.POINTER(ctypes.c_double_complex))]
        
        rho_struct = ComplexArray_c_ptr(N * N, rho_ptr)
        
        # The structure of the C function call requires proper pointer handling
        # For simplicity, we assume compute_purity expects the array data pointer and size
        # A more robust solution involves creating a proper wrapper class
        
        # Fallback to direct call with pointer if structure is too complex for quick testing
        try:
             # A simpler call assuming C function takes a pointer to the data and the size
             purity = C_CORE.compute_purity(ctypes.cast(rho_ptr, ctypes.c_void_p), N)
        except Exception as e:
             print(f"C-Core function call failed: {e}. Falling back to mock purity.")
             purity = 0.5 + np.random.rand() * 0.5 # Mock Purity calculation
    else:
        purity = 0.5 + np.random.rand() * 0.5 # Mock Purity calculation

    # 3. Decision/Mapping (Control Engine)
    current_intent = "increase_performance" if purity < 0.6 else "defensive_stance_disconnect"
    CONTROL_MAPPER.map_intent_to_action(current_intent)

    # 4. Update Global State
    SYSTEM_STATE["system_purity"] = purity
    SYSTEM_STATE["emotional_valence"] += (purity - 0.7) * 0.1
    SYSTEM_STATE["llm_summary"] = f"Purity: {purity:.4f}. Intent: {current_intent}. Sensor Grad: {sensor_update[0]:.2f}."


    return {
        "status": "step_complete",
        "system_state": SYSTEM_STATE,
        "geometric_update": sensor_update,
        "purity_tr_rho_sq": purity,
    }

