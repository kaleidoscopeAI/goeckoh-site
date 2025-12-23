pub fn contractive_update(&mut self, weights: [[f64; N]; N]) {
    for i in 0..N {
        let mut sum = 0.0;
        for j in 0..N {
            sum += weights[i][j] * (self.values[j].tanh());
        }
        self.values[i] = 0.95 * self.values[i] + 0.05 * sum;
    }
}
