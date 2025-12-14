use std::hint::black_box;
use std::path::PathBuf;
use std::time::Instant;

use rust_het_graph::alg::bfs_coop::CoopBfsPlan;
use rust_het_graph::graph::compare::summarize_vs_cggraph;
use rust_het_graph::graph::fps::{print_speed_summary, read_speed_records, run_time_sanity};
use rust_het_graph::graph::io_bin::load_reordered_csr_from_dir;
use rust_het_graph::graph::stats::validate_csr;
use rust_het_graph::graph::subgraph::SubgraphPolicy;

fn usage_and_exit() -> ! {
    eprintln!(
        r#"Usage:
  cargo run --release --bin compare_bfs_twitter_partsmall -- \
    <reordered_bin_dir> [root_new] [edge_budget] [per_vertex_cap] \
    [--policy <vertex-prefix|adj-prefix>] \
    [--trials N] [--quiet] \
    [--cg-cpu <FPS_CPU_*.bin>] [--cg-gpu <FPS_GPU_*.bin>]

Defaults:
  root_new       = 0
  edge_budget    = 1500000
  per_vertex_cap = 4096
  policy         = vertex-prefix
  trials         = 7
  debug          = on
"#
    );
    std::process::exit(2);
}

fn median_ms(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mid = xs.len() / 2;
    if xs.len() % 2 == 1 {
        xs[mid]
    } else {
        0.5 * (xs[mid - 1] + xs[mid])
    }
}

fn parse_args() -> (
    PathBuf,
    u32,
    usize,
    usize,
    SubgraphPolicy,
    usize,
    bool,
    Option<PathBuf>,
    Option<PathBuf>,
) {
    let mut args = std::env::args().skip(1);

    let dir = args.next().map(PathBuf::from).unwrap_or_else(|| usage_and_exit());

    let root_new: u32 = args
        .next()
        .as_deref()
        .unwrap_or("0")
        .parse()
        .unwrap_or_else(|_| usage_and_exit());

    let edge_budget: usize = args
        .next()
        .as_deref()
        .unwrap_or("1500000")
        .parse()
        .unwrap_or_else(|_| usage_and_exit());

    let per_vertex_cap: usize = args
        .next()
        .as_deref()
        .unwrap_or("4096")
        .parse()
        .unwrap_or_else(|_| usage_and_exit());

    let mut policy = SubgraphPolicy::VertexPrefix;
    let mut trials: usize = 7;
    let mut debug: bool = true;
    let mut cg_cpu: Option<PathBuf> = None;
    let mut cg_gpu: Option<PathBuf> = None;

    while let Some(tok) = args.next() {
        match tok.as_str() {
            "--policy" => {
                let val = args.next().unwrap_or_else(|| usage_and_exit());
                policy = SubgraphPolicy::parse(&val).unwrap_or_else(|| usage_and_exit());
            }
            "--trials" => {
                let val = args.next().unwrap_or_else(|| usage_and_exit());
                trials = val.parse().unwrap_or_else(|_| usage_and_exit());
                if trials == 0 {
                    eprintln!("--trials must be >= 1");
                    std::process::exit(2);
                }
            }
            "--quiet" => {
                debug = false;
            }
            "--cg-cpu" => {
                let val = args.next().unwrap_or_else(|| usage_and_exit());
                cg_cpu = Some(PathBuf::from(val));
            }
            "--cg-gpu" => {
                let val = args.next().unwrap_or_else(|| usage_and_exit());
                cg_gpu = Some(PathBuf::from(val));
            }
            _ => usage_and_exit(),
        }
    }

    (dir, root_new, edge_budget, per_vertex_cap, policy, trials, debug, cg_cpu, cg_gpu)
}

fn main() -> cust::error::CudaResult<()> {
    let (dir, root_new, edge_budget, per_vertex_cap, policy, trials, debug, cg_cpu, cg_gpu) =
        parse_args();

    println!("[INPUT] reordered_bin_dir = {:?}", dir);
    println!("[INPUT] root_new          = {}", root_new);
    println!("[INPUT] edge_budget       = {}", edge_budget);
    println!("[INPUT] per_vertex_cap    = {}", per_vertex_cap);
    println!("[INPUT] policy            = {:?}", policy);
    println!("[INPUT] trials            = {}", trials);
    println!("[INPUT] debug             = {}", debug);

    let g = load_reordered_csr_from_dir(&dir).unwrap_or_else(|e| {
        eprintln!("Load reordered CSR failed: {e}");
        std::process::exit(1);
    });

    validate_csr(&g).unwrap_or_else(|e| {
        eprintln!("CSR validation failed: {e}");
        std::process::exit(1);
    });

    println!("\n[GRAPH] n={}, m={}", g.n(), g.m());

    println!("\n[PLAN] Building + uploading GPU subgraph once ...");
    let mut plan = CoopBfsPlan::new(&g, edge_budget, per_vertex_cap, policy)?;

    println!("\n[RUN] Warm-up coop BFS (plan reuse) ...");
    let warm = plan.run(root_new, false)?;
    black_box(&warm);

    println!("\n[RUN] Timed coop BFS: {} trials (median) ...", trials);
    let mut times_ms: Vec<f64> = Vec::with_capacity(trials);
    let mut last = None;

    for t in 0..trials {
        let t0 = Instant::now();
        let res = plan.run(root_new, debug)?;
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        times_ms.push(ms);
        last = Some(res);

        println!(
            "[RUN] trial {}: {:.3} ms (reached={}, maxd={})",
            t,
            ms,
            last.as_ref().unwrap().reached,
            last.as_ref().unwrap().maxd
        );
    }

    let mut sorted = times_ms.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = median_ms(times_ms);

    let last = last.expect("must exist because trials>=1");
    println!("\n[RESULT] coop BFS times ms (sorted): {:?}", sorted);
    println!(
        "[RESULT] coop BFS min/median/max = {:.3} / {:.3} / {:.3} ms",
        sorted[0],
        med,
        sorted[sorted.len() - 1]
    );
    println!(
        "[RESULT] coop BFS reached {} / {} vertices, maxd={}, levels={}",
        last.reached,
        g.n(),
        last.maxd,
        last.levels
    );
    println!(
        "[RESULT] subgraph: policy={:?}, m_gpu={}, edges_on_gpu={:.4}",
        last.policy, last.m_gpu, last.edges_on_gpu
    );

    if let Some(cg_cpu_path) = cg_cpu.as_ref() {
        println!("\n[CG] Reading FPS (CPU) from {:?}", cg_cpu_path);
        let runs = read_speed_records(cg_cpu_path.to_string_lossy().as_ref()).unwrap_or_else(|e| {
            eprintln!("read_speed_records (CPU) failed: {e}");
            std::process::exit(1);
        });

        println!("[CG] Sanity check (CPU): sum(time_ms) vs stored_total_ms");
        for (i, run) in runs.iter().enumerate() {
            let (sum_ms, stored_ms) = run_time_sanity(run);
            println!(
                "  CPU run {}: sum(time_ms)={:.3} ms | stored_total={:.3} ms | diff={:.3} ms",
                i,
                sum_ms,
                stored_ms,
                stored_ms - sum_ms
            );
        }

        print_speed_summary("CG BFS (CPU)", &runs, 20);
        summarize_vs_cggraph("BFS vs CG CPU", med, last.reached, format!("{}", last.maxd), &runs);
    } else {
        println!("\n[CG] --cg-cpu not provided; skipping CG CPU comparison.");
    }

    if let Some(cg_gpu_path) = cg_gpu.as_ref() {
        println!("\n[CG] Reading FPS (GPU) from {:?}", cg_gpu_path);
        let runs = read_speed_records(cg_gpu_path.to_string_lossy().as_ref()).unwrap_or_else(|e| {
            eprintln!("read_speed_records (GPU) failed: {e}");
            std::process::exit(1);
        });

        println!("[CG] Sanity check (GPU): sum(time_ms) vs stored_total_ms");
        for (i, run) in runs.iter().enumerate() {
            let (sum_ms, stored_ms) = run_time_sanity(run);
            println!(
                "  GPU run {}: sum(time_ms)={:.3} ms | stored_total={:.3} ms | diff={:.3} ms",
                i,
                sum_ms,
                stored_ms,
                stored_ms - sum_ms
            );
        }

        print_speed_summary("CG BFS (GPU)", &runs, 20);
        summarize_vs_cggraph("BFS vs CG GPU", med, last.reached, format!("{}", last.maxd), &runs);
    } else {
        println!("[CG] --cg-gpu not provided; skipping CG GPU comparison.");
    }

    Ok(())
}
