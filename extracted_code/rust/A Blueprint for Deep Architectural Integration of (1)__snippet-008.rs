// File: benches/forward_step_bench.rs
use criterion::{criterion_group, criterion_main, Criterion, black_box};
use rand_distr::{Normal, Distribution};
use rand::thread_rng;

type Vector = Vec<f64>;

struct Engine {
    params: Vector,
    operator: Box<dyn Fn(&Vector, &Vector) -> Vector>,
}

struct CognitiveSystem {
    engines: Vec<Engine>,
    routing: Vec<Vec<f64>>,
    global_state: Vector,
    threshold: f64,
    noise_level: f64,
}

impl CognitiveSystem {
    fn forward_step(&mut self, external_input: &Vector) {
        let n = self.engines.len();
        let mut outputs = vec![vec![]; n];

        // Compute outputs per engine
        for i in 0..n {
            let input_i = self.compose_input(i);
            outputs[i] = (self.engines[i].operator)(&input_i, &self.engines[i].params);
        }

        // Aggregate weighted outputs into new state
        let mut new_state = vec![0.0; self.global_state.len()];
        for i in 0..n {
            for (&val, ns_val) in outputs[i].iter().zip(new_state.iter_mut()) {
                *ns_val += val * self.routing[i][i]; // simplified weighting
            }
        }

        // Add external input
        for (ns_val, ext_val) in new_state.iter_mut().zip(external_input.iter()) {
            *ns_val += *ext_val;
        }

        // Add Gaussian noise
        let normal = Normal::new(0.0, self.noise_level).unwrap();
        let mut rng = thread_rng();
        for val in new_state.iter_mut() {
            *val += normal.sample(&mut rng);
        }

        self.global_state = new_state;
    }

    fn compose_input(&self, i: usize) -> Vector {
        let mut input = Vec::new();

        // Project global state (clip for example)
        input.extend(self.global_state.iter());

        // Append connected engine outputs (empty for benchmark simplicity)
        input
    }
}

fn bench_forward_step(c: &mut Criterion) {
    let n_engines = 5;
    let mut system = CognitiveSystem {
        engines: (0..n_engines)
            .map(|_| Engine {
                params: vec![0.1; 10],
                operator: Box::new(|input, params| {
                    input.iter().zip(params.iter().cycle()).map(|(x, p)| x * p).collect()
                }),
            })
            .collect(),
        routing: vec![vec![1.0; n_engines]; n_engines],
        global_state: vec![0.5; 20],
        threshold: 0.05,
        noise_level: 0.01,
    };

    let ext_input = vec![0.1; 20];

    c.bench_function("cognitive_forward_step", |b| {
        b.iter(|| {
            system.forward_step(black_box(&ext_input));
        })
    });
}

criterion_group!(benches, bench_forward_step);
criterion_main!(benches);
