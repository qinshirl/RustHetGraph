// Run with:
//   BIN_DIR=/data/webgraph/bin/web-Google ROOT_NEW=0 \
//   EDGE_BUDGET=1500000 PER_VERTEX_CAP=4096 \
//   cargo bench --bench pipeline

use std::path::PathBuf;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use rust_het_graph::alg::bfs::bfs;
use rust_het_graph::alg::bfs_coop::bfs_coop_gpu_cpu;
use rust_het_graph::graph::io_bin::load_reordered_csr_from_dir;
use rust_het_graph::graph::stats::validate_csr;
use rust_het_graph::graph::subgraph::SubgraphPolicy;

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn env_u32(key: &str, default: u32) -> u32 {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn env_bool(key: &str, default: bool) -> bool {
    match std::env::var(key).ok().as_deref() {
        Some("1") | Some("true") | Some("TRUE") | Some("yes") | Some("YES") => true,
        Some("0") | Some("false") | Some("FALSE") | Some("no") | Some("NO") => false,
        Some(_) => default,
        None => default,
    }
}

fn pipeline_benchmark(c: &mut Criterion) {

    let bin_dir = std::env::var("BIN_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/data/webgraph/bin/web-Google"));

    let root_new: u32 = env_u32("ROOT_NEW", 0);
    let edge_budget: usize = env_usize("EDGE_BUDGET", 1_500_000);
    let per_vertex_cap_vp: usize = env_usize("PER_VERTEX_CAP", 4096); // used for vertex-prefix call (ignored by policy anyway)
    let per_vertex_cap_ap: usize = env_usize("PER_VERTEX_CAP_ADJ", 64); // used for adj-prefix
    let coop_debug: bool = env_bool("COOP_DEBUG", false);

    // load once
    let g = load_reordered_csr_from_dir(&bin_dir).unwrap_or_else(|e| {
        panic!("load_reordered_csr_from_dir failed for {:?}: {}", bin_dir, e);
    });

    validate_csr(&g).unwrap_or_else(|e| {
        panic!("CSR validation failed: {}", e);
    });

    let n = g.n();
    let m = g.m();


    let mut group = c.benchmark_group("pipeline_bfs");
    group.sample_size(10);

    // cpu bfs baseline
    group.bench_with_input(
        BenchmarkId::new("cpu_bfs", format!("n={}_m={}_root={}", n, m, root_new)),
        &root_new,
        |b, &root| {
            b.iter(|| {
                let dist = bfs(&g, root);
                black_box(dist);
            })
        },
    );

    //vertex-prefix
    group.bench_with_input(
        BenchmarkId::new(
            "coop_bfs_vertex_prefix",
            format!(
                "n={}_m={}_root={}_budget={}_cap={}",
                n, m, root_new, edge_budget, per_vertex_cap_vp
            ),
        ),
        &root_new,
        |b, &root| {
            b.iter(|| {
                let res = bfs_coop_gpu_cpu(
                    &g,
                    root,
                    edge_budget,
                    per_vertex_cap_vp,
                    SubgraphPolicy::VertexPrefix,
                    coop_debug,
                )
                .expect("bfs_coop_gpu_cpu (vertex-prefix) failed");
                // prevent optimization
                black_box(res.dist);
            })
        },
    );

    // adj-prefix
    group.bench_with_input(
        BenchmarkId::new(
            "coop_bfs_adj_prefix",
            format!(
                "n={}_m={}_root={}_budget={}_cap={}",
                n, m, root_new, edge_budget, per_vertex_cap_ap
            ),
        ),
        &root_new,
        |b, &root| {
            b.iter(|| {
                let res = bfs_coop_gpu_cpu(
                    &g,
                    root,
                    edge_budget,
                    per_vertex_cap_ap,
                    SubgraphPolicy::AdjPrefix,
                    coop_debug,
                )
                .expect("bfs_coop_gpu_cpu (adj-prefix) failed");
                black_box(res.dist);
            })
        },
    );

    group.finish();
}

criterion_group!(benches, pipeline_benchmark);
criterion_main!(benches);
