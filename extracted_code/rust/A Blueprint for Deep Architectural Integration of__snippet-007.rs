use criterion::{criterion_group, criterion_main, Criterion};

fn bench_forward_step(c: &mut Criterion) {
    c.bench_function("forward_step", |b| {
        let mut system = /* initialize your CognitiveSystem */;
        let input = vec![0.1; GLOBAL_STATE_SIZE];
        b.iter(|| system.forward_step(&input));
    });
}

criterion_group!(benches, bench_forward_step);
criterion_main!(benches);
