• I'm prioritizing using the Rust lattice via an optional env var flag while
  keeping the Python lattice for now due to FFI limits, and updating the
  validation script to conditionally import matplotlib to avoid failures when
  plotting isn't required.

