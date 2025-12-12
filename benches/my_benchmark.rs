use criterion::{criterion_group, criterion_main, Criterion};
use petgraph::algo::dijkstra;
use rust_het_graph::{convert_graph_2_index_weight, load_graph_from_neo4j};

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("convert_graph_2_index_weight",
                     |b| {
                         let rt  = tokio::runtime::Runtime::new().unwrap();
                         let graph = rt.block_on(load_graph_from_neo4j());
                         let graph = convert_graph_2_index_weight(graph);
                         b.iter(|| {
                             let idx = graph.node_indices().nth(0).unwrap();
                             dijkstra(&graph, idx, None, |e| *e.weight());
                         })
                     });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
