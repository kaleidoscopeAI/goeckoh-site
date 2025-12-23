let sum: f32 = buf.iter().copied().sum();
let mut freq_mhz = 300f32 + (sum.fract() * 1700f32); // 300..2000 MHz
