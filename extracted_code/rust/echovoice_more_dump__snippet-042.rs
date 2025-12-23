pub fn new(confidence: f64, can_process: bool, quality: f64) -> Self {
    Self {
        confidence: confidence.clamp(0.0, 1.0),
        can_process,
        estimated_quality: quality.clamp(0.0, 1.0),
    }
}

pub fn cannot_process() -> Self {
    Self::new(0.0, false, 0.0)
}

pub fn perfect() -> Self {
    Self::new(1.0, true, 1.0)
}
