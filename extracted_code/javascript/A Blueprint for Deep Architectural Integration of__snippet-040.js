let activated = tanh_activation(x);
let wx: Vec<f64> = W.iter()
    .map(|row| row.iter().zip(activated.iter()).map(|(w, a)| w * a).sum())
    .collect();
wx.iter().zip(b.iter()).map(|(v, b)| v + b).collect()
