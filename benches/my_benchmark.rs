use criterion::{criterion_group, criterion_main, Criterion};
use rust_het_graph::{convert_graph_2_index_weight, load_graph_from_neo4j};

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("convert_graph_2_index_weight",
                     |b| {b.iter(async || convert_graph_2_index_weight(load_graph_from_neo4j().await))});
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
