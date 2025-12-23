let mut count = 0;
let mut val = state;
while val != 0 {
    count += val & 1;
    val >>= 1;
}
(count * 100 / 64) as u64  // Normalized 0-100
