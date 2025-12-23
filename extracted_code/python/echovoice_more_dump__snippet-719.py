    from cso_core.python_fallback import SwarmPython  # Assumed Python class
    
    SwarmOptimizer = SwarmPython
    
    # Log a performance warning
    logger.warning("C/Assembly extension (cso_fast) not found. Falling back to pure Python (SwarmPython). Expect degraded performance.")

