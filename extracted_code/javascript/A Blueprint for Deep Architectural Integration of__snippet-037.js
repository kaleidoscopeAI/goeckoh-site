c.bench_function("forward_step", |b| {
    let mut system = /* initialize your CognitiveSystem */;
    let input = vec![0.1; GLOBAL_STATE_SIZE];
    b.iter(|| system.forward_step(&input));
});
