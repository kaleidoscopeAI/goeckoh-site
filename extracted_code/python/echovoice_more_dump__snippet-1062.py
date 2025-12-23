# (D) Attempt to import the compiled C/Assembly extension
from cso_fast import SwarmFast

# Success: Use the fast implementation
SwarmOptimizer = SwarmFast

# Log a high-priority message for confirmation
logger.info("Successfully loaded compiled C/Assembly extension (cso_fast). Running in MAX_PERF mode.")

