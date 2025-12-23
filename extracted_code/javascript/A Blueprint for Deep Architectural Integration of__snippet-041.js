let W = vec![vec![0.3, 0.2], vec![0.1, 0.4]];
let b = vec![0.1, -0.1];
let mut x = vec![0.0, 0.0];
let mut y = vec![1.0, 1.0];

for _ in 0..20 {
    x = update(&x, &W, &b);
    y = update(&y, &W, &b);
    let dist = euclidean_distance(&x, &y);
    println!("Distance: {}", dist); // exponentially decreases
}
