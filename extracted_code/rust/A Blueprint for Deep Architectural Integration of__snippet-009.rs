fn tanh_activation(v: &Vec<f64>) -> Vec<f64> {
    v.iter().map(|x| x.tanh()).collect()
}

fn update(x: &Vec<f64>, W: &Vec<Vec<f64>>, b: &Vec<f64>) -> Vec<f64> {
    let activated = tanh_activation(x);
    let wx: Vec<f64> = W.iter()
        .map(|row| row.iter().zip(activated.iter()).map(|(w, a)| w * a).sum())
        .collect();
    wx.iter().zip(b.iter()).map(|(v, b)| v + b).collect()
}

fn simulate() {
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
}
