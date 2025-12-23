fn clamp(x: f32, lo: f32, hi: f32) -> f32 { x.max(lo).min(hi) }
fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }

