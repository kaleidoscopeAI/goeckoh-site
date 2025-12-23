let mut outv = Vec::with_capacity(3);
let t0 = buf.iter().fold(0f32, |a,b| a + b);
let aw = ((t0 * 12.9898).sin() * 43758.5453).abs() % 1.0;
let va = ((t0 * 78.233).cos() * 12345.678).abs() % 1.0;
let coh = 0.6 + ((t0 * 0.37).sin() * 0.4).abs();
