pub fn update_curiosity_bit(&mut self, perf: f32, ext_drive: f32) {
    let input = self.rho * if self.curiosity_bit { self.lambda } else { 0.0 }
        - self.sigma * perf
        + self.beta * ext_drive;
    self.curiosity_bit = input > self.theta;
