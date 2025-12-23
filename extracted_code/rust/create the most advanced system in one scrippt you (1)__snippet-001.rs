#![feature(generic_const_exprs)]
use verus::*;
use coq_of_rust::*;
use rus_synth::*;
use contraction_nn::*;
use dyn_robust::*;

struct StateVector<const N: usize> {
    values: [f64; N],
}

impl<const N: usize> StateVector<N> {
    pub fn contractive_update(&mut self, weights: [[f64; N]; N]) {
        for i in 0..N {
            let mut sum = 0.0;
            for j in 0..N {
                sum += weights[i][j] * (self.values[j].tanh());
            }
            self.values[i] = 0.95 * self.values[i] + 0.05 * sum;
        }
    }
}

verus_verify! {
    pub const fn stable_contraction<const N: usize>(s: StateVector<N>, w: [[f64; N]; N])
        requires
            symmetric(w),
            trace_lt_1(w),
        ensures
            contractive(s, w),
    { }
}

fn main() {
    let mut system = StateVector::<8> { values: [0.0; 8] };
    let weights = synthesize_contractive_matrix(8);
    for _ in 0..1000 { system.contractive_update(weights); }
    assert!(system.values.iter().all(|v| v.abs() < 1.0));
}
