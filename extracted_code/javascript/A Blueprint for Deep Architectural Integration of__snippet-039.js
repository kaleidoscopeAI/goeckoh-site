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
