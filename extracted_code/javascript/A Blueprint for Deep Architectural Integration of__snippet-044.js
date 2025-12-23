let w = vec![vec![0.3, 0.2], vec![0.1, 0.4]];
let b = vec![0.1, -0.1];
let mut x1 = vec![0.0, 0.0];
let mut x2 = vec![1.0, 1.0];

for _ in 0..30 {
    x1 = step(&x1, &w, &b);
    x2 = step(&x2, &w, &b);
    let dist = x1.iter().zip(x2.iter()).map(|(a, b)| (a-b).powi(2)).sum::<f64>().sqrt();
    println!("Distance: {}", dist);
}
