use rand_distr::{Normal, Distribution};
let normal = Normal::new(0.0, noise_level).unwrap();
for val in new_state.iter_mut() {
    *val += normal.sample(&mut rand::thread_rng());
}
