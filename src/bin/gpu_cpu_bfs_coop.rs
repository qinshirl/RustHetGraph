// use cust::prelude::*;
// use rayon::prelude::*;
// use std::path::PathBuf;
// use std::sync::atomic::{AtomicU8, Ordering};
// use std::time::Instant;

// use rust_het_graph::gpu::runtime::FRONTIER_EXPAND_PTX;
// use rust_het_graph::graph::io_bin::load_reordered_csr_from_dir;
// use rust_het_graph::graph::stats::validate_csr;
// use rust_het_graph::graph::subgraph::{build_gpu_subgraph, GpuSubgraph, SubgraphPolicy};

// fn bfs_stats(dist: &[i32]) -> (u64, i32) {
//     let mut reached: u64 = 0;
//     let mut maxd: i32 = -1;
//     for &d in dist {
//         if d >= 0 {
//             reached += 1;
//             if d > maxd {
//                 maxd = d;
//             }
//         }
//     }
//     (reached, maxd)
// }

// fn usage_and_exit() -> ! {
//     eprintln!(
//         r#"Usage:
//   cargo run --release --bin gpu_cpu_bfs_coop -- <bin_dir> [root_new] [edge_budget] [per_vertex_cap] [--policy <vertex-prefix|adj-prefix>]

// Notes:
//   - Default policy is vertex-prefix (CGgraph-style contiguous vertex cutoff).
//   - per_vertex_cap is only meaningful for adj-prefix. It is ignored for vertex-prefix.

// Examples:
//   # CGgraph mimic(vertex prefix)
//   cargo run --release --bin gpu_cpu_bfs_coop -- /data/webgraph/bin/web-Google 0 1500000 4096 --policy vertex-prefix

//   # Fine-grained budgeting (adjacency prefix per vertex)
//   cargo run --release --bin gpu_cpu_bfs_coop -- /data/webgraph/bin/web-Google 0 1500000 64 --policy adj-prefix
// "#
//     );
//     std::process::exit(2);
// }

// /// CLI parser:
// /// positional: bin_dir, root_new, edge_budget, per_vertex_cap
// /// flag: --policy <vertex-prefix|adj-prefix>
// fn parse_args() -> (PathBuf, u32, usize, usize, SubgraphPolicy) {
//     let mut args = std::env::args().skip(1);

//     let out_dir = args.next().map(PathBuf::from).unwrap_or_else(|| usage_and_exit());

//     let root_new: u32 = args
//         .next()
//         .as_deref()
//         .unwrap_or("0")
//         .parse()
//         .unwrap_or_else(|_| usage_and_exit());

//     let edge_budget: usize = args
//         .next()
//         .as_deref()
//         .unwrap_or("1500000")
//         .parse()
//         .unwrap_or_else(|_| usage_and_exit());

//     let per_vertex_cap: usize = args
//         .next()
//         .as_deref()
//         .unwrap_or("4096")
//         .parse()
//         .unwrap_or_else(|_| usage_and_exit());

//     // Default policy: CGgraph-like
//     let mut policy = SubgraphPolicy::VertexPrefix;

//     while let Some(tok) = args.next() {
//         if tok == "--policy" {
//             let val = args.next().unwrap_or_else(|| usage_and_exit());
//             policy = SubgraphPolicy::parse(&val).unwrap_or_else(|| usage_and_exit());
//         } else {
//             usage_and_exit();
//         }
//     }

//     (out_dir, root_new, edge_budget, per_vertex_cap, policy)
// }

// fn main() -> cust::error::CudaResult<()> {
//     let (out_dir, root_new, edge_budget, per_vertex_cap, policy) = parse_args();

//     // Load reordered CSR
//     let g = load_reordered_csr_from_dir(&out_dir).unwrap();
//     validate_csr(&g).unwrap();

//     let n: usize = g.n();
//     let m: usize = g.m();

//     println!("Graph: n={}, m={}", n, m);
//     println!("root_new={}", root_new);
//     println!("GPU edge_budget={}", edge_budget);
//     println!("GPU per_vertex_cap={}", per_vertex_cap);
//     println!("policy={:?}", policy);

//     // Build GPU subgraph under selected policy
//     let sub: GpuSubgraph = build_gpu_subgraph(&g, policy, edge_budget, per_vertex_cap);
//     let m_gpu = sub.m_gpu();

//     // Global metric
//     let edges_on_gpu = if m == 0 { 0.0 } else { (m_gpu as f64) / (m as f64) };

//     match policy {
//         SubgraphPolicy::VertexPrefix => {
//             let cut = sub.cut_vertex_opt().unwrap_or(0);
//             println!(
//                 "Built G': cut_vertex={} m_gpu={} / m={} => edges_on_gpu={:.4}",
//                 cut, m_gpu, m, edges_on_gpu
//             );
//         }
//         SubgraphPolicy::AdjPrefix => {
//             println!(
//                 "Built G': m_gpu={} / m={} => edges_on_gpu={:.4}",
//                 m_gpu, m, edges_on_gpu
//             );
//         }
//     }

//     // CUDA init + module load
//     let _ctx = cust::quick_init()?;
//     let module = Module::from_ptx(FRONTIER_EXPAND_PTX, &[])?;
//     let k = module.get_function("frontier_expand_subgraph_only")?;
//     let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

//     // Upload GPU CSR to device
//     let d_gpu_offsets = DeviceBuffer::from_slice(sub.gpu_offsets())?;
//     let d_gpu_dst = DeviceBuffer::from_slice(sub.gpu_dst())?;

//     // Per-iteration GPU output buffers (conservative n-cap; guarded in kernel by out_cap)
//     let d_out = DeviceBuffer::<u32>::zeroed(n)?;
//     let mut d_write_count = DeviceBuffer::from_slice(&[0u32])?;

//     // Authoritative visited/dist on CPU
//     let visited: Vec<AtomicU8> = (0..n).map(|_| AtomicU8::new(0)).collect();
//     visited[root_new as usize].store(1, Ordering::Release);

//     let mut dist: Vec<i32> = vec![-1; n];
//     dist[root_new as usize] = 0;

//     let mut frontier: Vec<u32> = vec![root_new];
//     let mut level: i32 = 0;

//     let t_total = Instant::now();

//     while !frontier.is_empty() {
//         // Partition frontier into:
//         // - frontier_gpu: vertices whose edges are available on GPU (policy-specific definition)
//         // - frontier_cpu_only: vertices that must be expanded on CPU only
//         let mut frontier_cpu_only: Vec<u32> = Vec::new();
//         let mut frontier_gpu: Vec<u32> = Vec::new();

//         let gpu_offsets = sub.gpu_offsets();
//         let cut_vertex = sub.cut_vertex_opt(); // Some(_) only for VertexPrefix

//         for &u in &frontier {
//             let ui = u as usize;

//             // Defensive: invalid ids -> CPU-only (and will be discarded later)
//             if ui + 1 >= gpu_offsets.len() {
//                 frontier_cpu_only.push(u);
//                 continue;
//             }

//             let on_gpu = match policy {
//                 // CGgraph-style: membership is by contiguous vertex id range
//                 SubgraphPolicy::VertexPrefix => cut_vertex.map(|c| u < c).unwrap_or(false),

//                 // AdjPrefix: membership is whether the vertex has any GPU prefix edges
//                 SubgraphPolicy::AdjPrefix => {
//                     let gpu_deg = gpu_offsets[ui + 1] - gpu_offsets[ui];
//                     gpu_deg > 0
//                 }
//             };

//             if on_gpu {
//                 frontier_gpu.push(u);
//             } else {
//                 frontier_cpu_only.push(u);
//             }
//         }

//         // Per-level metric: frontier_on_gpu
//         let frontier_on_gpu = (frontier_gpu.len() as f64) / (frontier.len() as f64);

//         // Debug counters (good for “work split” discussion)
//         let mut cpu_only_edges: u64 = 0;
//         let mut gpu_edges_expected: u64 = 0;
//         let mut cpu_residual_edges: u64 = 0;

//         for &u in &frontier_cpu_only {
//             let ui = u as usize;
//             if ui + 1 < g.offsets.len() {
//                 cpu_only_edges += (g.offsets[ui + 1] - g.offsets[ui]) as u64;
//             }
//         }
//         for &u in &frontier_gpu {
//             let ui = u as usize;
//             if ui + 1 < gpu_offsets.len() {
//                 gpu_edges_expected += (gpu_offsets[ui + 1] - gpu_offsets[ui]) as u64;
//             }
//         }

//         // Residual edges only exist under adj-prefix
//         if policy == SubgraphPolicy::AdjPrefix {
//             if let Some(gpu_edge_cut) = sub.gpu_edge_cut_opt() {
//                 for &u in &frontier_gpu {
//                     let ui = u as usize;
//                     if ui + 1 < g.offsets.len() && ui < gpu_edge_cut.len() {
//                         let end = g.offsets[ui + 1];
//                         let cut = gpu_edge_cut[ui];
//                         if cut < end {
//                             cpu_residual_edges += (end - cut) as u64;
//                         }
//                     }
//                 }
//             }
//         }

//         // GPU launch (async): expand GPU edges for frontier_gpu
//         let mut gpu_candidates: Vec<u32> = Vec::new();

//         if !frontier_gpu.is_empty() {
//             d_write_count.copy_from(&[0u32])?;
//             let d_frontier = DeviceBuffer::from_slice(&frontier_gpu)?;

//             let frontier_len = frontier_gpu.len() as u32;
//             let threads = 256u32;
//             let blocks = ((frontier_len + threads - 1) / threads).max(1);
//             let out_cap = d_out.len() as u32;

//             unsafe {
//                 launch!(
//                     k<<<blocks, threads, 0, stream>>>(
//                         d_gpu_offsets.as_device_ptr(),
//                         d_gpu_dst.as_device_ptr(),
//                         d_frontier.as_device_ptr(),
//                         d_out.as_device_ptr(),
//                         d_write_count.as_device_ptr(),
//                         frontier_len,
//                         out_cap
//                     )
//                 )?;
//             }
//             // no sync yet; overlap CPU below
//         }

//         // CPU work (parallel)
//         // CPU-only vertices expand FULL adjacency
//         let cpu_candidates_only: Vec<u32> = frontier_cpu_only
//             .par_iter()
//             .flat_map_iter(|&u| {
//                 let ui = u as usize;
//                 if ui + 1 >= g.offsets.len() {
//                     return Vec::new().into_iter();
//                 }
//                 let start = g.offsets[ui] as usize;
//                 let end = g.offsets[ui + 1] as usize;

//                 let mut local = Vec::with_capacity(end - start);
//                 for e in start..end {
//                     local.push(g.dst[e]);
//                 }
//                 local.into_iter()
//             })
//             .collect();

//         let mut cpu_candidates = cpu_candidates_only;

//         // adj-prefix only: GPU vertices expand RESIDUAL adjacency on CPU
//         if policy == SubgraphPolicy::AdjPrefix {
//             if let Some(gpu_edge_cut) = sub.gpu_edge_cut_opt() {
//                 let cpu_candidates_residual: Vec<u32> = frontier_gpu
//                     .par_iter()
//                     .flat_map_iter(|&u| {
//                         let ui = u as usize;
//                         if ui + 1 >= g.offsets.len() || ui >= gpu_edge_cut.len() {
//                             return Vec::new().into_iter();
//                         }
//                         let start = gpu_edge_cut[ui] as usize;
//                         let end = g.offsets[ui + 1] as usize;

//                         let mut local = Vec::with_capacity(end.saturating_sub(start));
//                         for e in start..end {
//                             local.push(g.dst[e]);
//                         }
//                         local.into_iter()
//                     })
//                     .collect();

//                 cpu_candidates.extend(cpu_candidates_residual);
//             }
//         }

//         // Synchronize and pull GPU results
//         if !frontier_gpu.is_empty() {
//             stream.synchronize()?;

//             let mut h_count = [0u32];
//             d_write_count.copy_to(&mut h_count)?;
//             let out_len = (h_count[0] as usize).min(d_out.len());

//             gpu_candidates.clear();
//             if out_len > 0 {
//                 let mut tmp = vec![0u32; d_out.len()];
//                 d_out.copy_to(&mut tmp)?;
//                 gpu_candidates.extend_from_slice(&tmp[..out_len]);
//             }
//         }

//         // DBUG 
//         if level <= 3 || (level % 3 == 0) {
//             println!(
//                 "[DEBUG] lvl={} frontier={} frontier_on_gpu={:.4} cpu_only_edges={} gpu_edges_exp={} cpu_residual_edges={} cpu_cand={} gpu_out={}",
//                 level,
//                 frontier.len(),
//                 frontier_on_gpu,
//                 cpu_only_edges,
//                 gpu_edges_expected,
//                 cpu_residual_edges,
//                 cpu_candidates.len(),
//                 gpu_candidates.len(),
//             );
//         } else {
//             println!(
//                 "[COOP BFS] level={} frontier={} frontier_on_gpu={:.4} cpu_only_frontier={} gpu_frontier={}",
//                 level,
//                 frontier.len(),
//                 frontier_on_gpu,
//                 frontier_cpu_only.len(),
//                 frontier_gpu.len(),
//             );
//         }

//         // merge candidates, authoritative visited/dist update on CPU
//         let mut merged = cpu_candidates;
//         merged.extend_from_slice(&gpu_candidates);

//         let mut next: Vec<u32> = Vec::new();
//         for v in merged {
//             let vi = v as usize;
//             if vi >= n {
//                 continue;
//             }

//             if visited[vi]
//                 .compare_exchange(0, 1, Ordering::AcqRel, Ordering::Relaxed)
//                 .is_ok()
//             {
//                 dist[vi] = level + 1;
//                 next.push(v);
//             }
//         }

//         println!(
//             "[COOP BFS] level={} next={} (cpu_only_frontier={}, gpu_frontier={})",
//             level,
//             next.len(),
//             frontier_cpu_only.len(),
//             frontier_gpu.len()
//         );

//         frontier = next;
//         level += 1;

//         if level > n as i32 {
//             break;
//         }
//     }

//     let ms = t_total.elapsed().as_secs_f64() * 1000.0;
//     let (reached, maxd) = bfs_stats(&dist);

//     println!(
//         "\n[COOP BFS] total_time={:.3} ms, reached={}, maxd={}",
//         ms, reached, maxd
//     );

//     Ok(())
// }
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// use std::path::PathBuf;

// use rust_het_graph::alg::bfs_coop::bfs_coop_gpu_cpu;
// use rust_het_graph::graph::io_bin::load_reordered_csr_from_dir;
// use rust_het_graph::graph::stats::validate_csr;
// use rust_het_graph::graph::subgraph::SubgraphPolicy;

// fn usage_and_exit() -> ! {
//     std::process::exit(2);
// }

// fn parse_args() -> (PathBuf, u32, usize, usize, SubgraphPolicy, bool) {
//     let mut args = std::env::args().skip(1);

//     let out_dir = args.next().map(PathBuf::from).unwrap_or_else(|| usage_and_exit());

//     let root_new: u32 = args
//         .next()
//         .as_deref()
//         .unwrap_or("0")
//         .parse()
//         .unwrap_or_else(|_| usage_and_exit());

//     let edge_budget: usize = args
//         .next()
//         .as_deref()
//         .unwrap_or("1500000")
//         .parse()
//         .unwrap_or_else(|_| usage_and_exit());

//     let per_vertex_cap: usize = args
//         .next()
//         .as_deref()
//         .unwrap_or("4096")
//         .parse()
//         .unwrap_or_else(|_| usage_and_exit());

//     // Defaults
//     let mut policy = SubgraphPolicy::VertexPrefix;
//     let mut debug = true;

//     while let Some(tok) = args.next() {
//         if tok == "--policy" {
//             let val = args.next().unwrap_or_else(|| usage_and_exit());
//             policy = SubgraphPolicy::parse(&val).unwrap_or_else(|_| usage_and_exit());
//         } else if tok == "--quiet" {
//             debug = false;
//         } else {
//             usage_and_exit();
//         }
//     }

//     (out_dir, root_new, edge_budget, per_vertex_cap, policy, debug)
// }

// fn main() -> cust::error::CudaResult<()> {
//     let (out_dir, root_new, edge_budget, per_vertex_cap, policy, debug) = parse_args();

//     // Load reordered CSR
//     let g = load_reordered_csr_from_dir(&out_dir).unwrap_or_else(|e| {
//         eprintln!("Load reordered CSR failed: {e}");
//         std::process::exit(1);
//     });

//     validate_csr(&g).unwrap_or_else(|e| {
//         eprintln!("CSR validation failed: {e}");
//         std::process::exit(1);
//     });

//     println!("Graph: n={}, m={}", g.n(), g.m());
//     println!("root_new={}", root_new);
//     println!("GPU edge_budget={}", edge_budget);
//     println!("GPU per_vertex_cap={}", per_vertex_cap);
//     println!("policy={:?}", policy);
//     println!("debug={}", debug);

//     let res = bfs_coop_gpu_cpu(&g, root_new, edge_budget, per_vertex_cap, policy, debug)?;
//     println!(
//         "\n[COOP BFS] total_time={:.3} ms, reached={}, maxd={}",
//         res.total_ms, res.reached, res.maxd
//     );

//     println!(
//         "[COOP BFS] policy={:?} m_gpu={} edges_on_gpu={:.4} levels={}",
//         res.policy, res.m_gpu, res.edges_on_gpu, res.levels
//     );

//     Ok(())
// }
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// src/bin/gpu_cpu_bfs_coop.rs

use std::path::PathBuf;

use cust::error::CudaResult;

use rust_het_graph::alg::bfs_coop::bfs_coop_gpu_cpu;
use rust_het_graph::graph::io_bin::load_reordered_csr_from_dir;
use rust_het_graph::graph::stats::validate_csr;
use rust_het_graph::graph::subgraph::SubgraphPolicy;

fn usage_and_exit() -> ! {
    eprintln!(
        r#"Usage:
  cargo run --release --bin gpu_cpu_bfs_coop -- <bin_dir> [root_new] [edge_budget] [per_vertex_cap]
      [--policy <vertex-prefix|adj-prefix>] [--quiet]

Positional args:
  bin_dir         Path to reordered CSR directory (contains csrOffset_u32.bin, csrDest_u32.bin, etc.)
  root_new        BFS root vertex id in the reordered (new) id space (default: 0)
  edge_budget     Max number of edges stored on GPU for the subgraph G' (default: 1500000)
  per_vertex_cap  For adj-prefix: cap on GPU edges per vertex (default: 4096)
                 For vertex-prefix: currently ignored by design (see notes below)

Flags:
  --policy <...>  Subgraph policy:
                    adj-prefix     (default) take a prefix of each vertex's adjacency list onto GPU, capped per vertex
                    vertex-prefix  take all edges for vertices [0..cut_vertex) onto GPU until budget is filled
  --quiet         Disable debug per-level prints

Notes:
  - In this codebase, per_vertex_cap is enforced only for adj-prefix.
  - If you run vertex-prefix, the GPU will only ever expand vertices with id < cut_vertex.
    If your BFS quickly leaves that id range, GPU contribution may drop to ~0 after the first hop.

Examples:
  # Recommended: cap enforced per vertex
  cargo run --release --bin gpu_cpu_bfs_coop -- /data/webgraph/bin/twitter-2010_partsmall_cg 0 1500000 4096 --policy adj-prefix

  # Vertex prefix: cap ignored, GPU covers only a small vertex-id prefix
  cargo run --release --bin gpu_cpu_bfs_coop -- /data/webgraph/bin/twitter-2010_partsmall_cg 0 1500000 4096 --policy vertex-prefix
"#
    );
    std::process::exit(2);
}

/// CLI parser:
/// positional: bin_dir, root_new, edge_budget, per_vertex_cap
/// flags: --policy <vertex-prefix|adj-prefix>, --quiet
fn parse_args() -> (PathBuf, u32, usize, usize, SubgraphPolicy, bool) {
    let mut args = std::env::args().skip(1);

    let bin_dir = args.next().map(PathBuf::from).unwrap_or_else(|| usage_and_exit());

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

    // IMPORTANT: default to AdjPrefix so per_vertex_cap actually means something.
    let mut policy = SubgraphPolicy::AdjPrefix;
    let mut debug = true;

    while let Some(tok) = args.next() {
        match tok.as_str() {
            "--policy" => {
                let val = args.next().unwrap_or_else(|| usage_and_exit());
                policy = SubgraphPolicy::parse(&val).unwrap_or_else(|| usage_and_exit());
            }
            "--quiet" => {
                debug = false;
            }
            _ => usage_and_exit(),
        }
    }

    (bin_dir, root_new, edge_budget, per_vertex_cap, policy, debug)
}

fn main() -> CudaResult<()> {
    let (bin_dir, root_new, edge_budget, per_vertex_cap, policy, debug) = parse_args();

    // Load reordered CSR
    let g = load_reordered_csr_from_dir(&bin_dir).unwrap();
    validate_csr(&g).unwrap();

    let n = g.n();
    let m = g.m();

    println!("Graph: n={}, m={}", n, m);
    println!("root_new={}", root_new);
    println!("GPU edge_budget={}", edge_budget);
    println!("GPU per_vertex_cap={}", per_vertex_cap);
    println!("policy={:?}", policy);
    println!("debug={}", debug);

    if policy == SubgraphPolicy::VertexPrefix {
        eprintln!(
            "[WARN] policy=VertexPrefix: per_vertex_cap is ignored by design in subgraph.rs. \
             If you want the cap enforced, use --policy adj-prefix."
        );
    }

    let res = bfs_coop_gpu_cpu(&g, root_new, edge_budget, per_vertex_cap, policy, debug)?;
    println!(
        "\n[COOP BFS] total_time={:.3} ms, reached={}, maxd={}",
        res.total_ms, res.reached, res.maxd
    );
    println!(
        "[COOP BFS] policy={:?} m_gpu={} edges_on_gpu={:.4} levels={}",
        res.policy, res.m_gpu, res.edges_on_gpu, res.levels
    );

    Ok(())
}
