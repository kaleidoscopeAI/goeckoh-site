    // R[i][j] = true if routing from engine j to i is enabled
    pub routing_matrix: Vec<Vec<bool>>,

    // Parameters as fixed constants for bit thresholds
    pub decay_rho: f32,
    pub lambda: f32,
    pub sigma: f32,
    pub beta: f32,
    pub threshold_theta: f32,
