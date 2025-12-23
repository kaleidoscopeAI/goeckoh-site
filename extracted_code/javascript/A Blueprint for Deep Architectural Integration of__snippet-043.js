let activated = tanh_vec(x);
let mut new_x = Vec::with_capacity(x.len());
for i in 0..x.len() {
    let sum: f64 = w[i].iter().zip(activated.iter()).map(|(wi, ai)| wi * ai).sum();
    new_x.push(sum + b[i]);
}
new_x
