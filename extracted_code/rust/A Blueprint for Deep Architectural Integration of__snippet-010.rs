fn tanh_vec(v: &Vec<f64>) -> Vec<f64> {
    v.iter().map(|x| x.tanh()).collect()
}

fn step(x: &Vec<f64>, w: &Vec<Vec<f64>>, b: &Vec<f64>) -> Vec<f64> {
    let activated = tanh_vec(x);
    let mut new_x = Vec::with_capacity(x.len());
    for i in 0..x.len() {
        let sum: f64 = w[i].iter().zip(activated.iter()).map(|(wi, ai)| wi * ai).sum();
        new_x.push(sum + b[i]);
    }
    new_x
}

fn simulate() {
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
}
