// State vector encoded as bits for n features
pub state_bits: Vec<u64>,   // each u64 holds 64 feature bits
pub curiosity_bit: bool,    // 1-bit curiosity activation

// Routing matrix stored as bitflags:
// R[i][j] = true if routing from engine j to i is enabled
pub routing_matrix: Vec<Vec<bool>>,

// Parameters as fixed constants for bit thresholds
pub decay_rho: f32,
pub lambda: f32,
pub sigma: f32,
pub beta: f32,
pub threshold_theta: f32,
