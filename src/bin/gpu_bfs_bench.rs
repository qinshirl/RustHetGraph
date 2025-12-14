use cust::prelude::*;
use rust_het_graph::gpu::runtime::FRONTIER_EXPAND_PTX;

use rust_het_graph::alg::bfs::bfs;
use rust_het_graph::graph::io_bin::{load_reordered_csr_from_dir, read_u32_bin};
use rust_het_graph::graph::stats::validate_csr;

use std::path::PathBuf;
use std::time::Instant;

fn median_ms(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mid = xs.len() / 2;
    if xs.len() % 2 == 1 {
        xs[mid]
    } else {
        0.5 * (xs[mid - 1] + xs[mid])
    }
}

fn bfs_stats(dist: &[i32]) -> (u64, i32) {
    let mut reached: u64 = 0;
    let mut maxd: i32 = -1;
    for &d in dist {
        if d >= 0 {
            reached += 1;
            if d > maxd {
                maxd = d;
            }
        }
    }
    (reached, maxd)
}

fn run_gpu_bfs_trial(
    stream: &Stream,
    kernel: &Function,
    d_offsets: &DeviceBuffer<u32>,
    d_dst: &DeviceBuffer<u32>,
    d_visited: &mut DeviceBuffer<u32>,
    d_dist: &mut DeviceBuffer<i32>,
    d_frontier_a: &mut DeviceBuffer<u32>,
    d_frontier_b: &mut DeviceBuffer<u32>,
    d_write_count: &mut DeviceBuffer<u32>,
    visited_init: &[u32],
    dist_init: &[i32],
    frontier_a_init: &[u32],
    root_new: u32,
    n: u32,
) -> cust::error::CudaResult<(f64, u64, i32)> {
    // Reset state (full-length copies; cust 0.3.x requires equal lengths)
    d_visited.copy_from(visited_init)?;
    d_dist.copy_from(dist_init)?;
    d_frontier_a.copy_from(frontier_a_init)?;
    d_write_count.copy_from(&[0u32])?;

    let threads: u32 = 256;
    let mut frontier_len: u32 = 1;
    let mut level: i32 = 0;

    let t0 = Instant::now();

    while frontier_len > 0 {
        // reset write_count
        d_write_count.copy_from(&[0u32])?;

        let blocks: u32 = (frontier_len + threads - 1) / threads;

        let stream = stream; // create an ident the macro can match
        unsafe {
            launch!(
                kernel<<<blocks, threads, 0, stream>>>(
                    d_offsets.as_device_ptr(),
                    d_dst.as_device_ptr(),
                    d_frontier_a.as_device_ptr(),
                    d_visited.as_device_ptr(),
                    d_dist.as_device_ptr(),
                    d_frontier_b.as_device_ptr(),
                    d_write_count.as_device_ptr(),
                    frontier_len,
                    level
                )
            )?;
        }


        stream.synchronize()?;

        // read next frontier length
        let mut h_count = [0u32];
        d_write_count.copy_to(&mut h_count)?;
        let next_len = h_count[0];

        // swap
        std::mem::swap(d_frontier_a, d_frontier_b);
        frontier_len = next_len;
        level += 1;

        if level > n as i32 {
            panic!("BFS level exceeded n; something is wrong.");
        }
    }

    let ms = t0.elapsed().as_secs_f64() * 1000.0;

    // Copy back dist and compute stats (for correctness guardrails)
    let mut dist_gpu: Vec<i32> = vec![0; n as usize];
    d_dist.copy_to(&mut dist_gpu)?;
    let (reached, maxd) = bfs_stats(&dist_gpu);

    // quick sanity: root distance is 0
    if dist_gpu[root_new as usize] != 0 {
        panic!("GPU dist[root_new] != 0 (got {})", dist_gpu[root_new as usize]);
    }

    Ok((ms, reached, maxd))
}

fn main() -> cust::error::CudaResult<()> {
    // cargo run --release --bin gpu_bfs_bench -- <bin_dir> [root_old] [trials]
    let out_dir = std::env::args().nth(1).map(PathBuf::from).unwrap_or_else(|| {
        eprintln!("Usage: cargo run --release --bin gpu_bfs_bench -- <bin_dir> [root_old] [trials]");
        std::process::exit(2);
    });

    let root_old: u32 = std::env::args()
        .nth(2)
        .as_deref()
        .unwrap_or("0")
        .parse()
        .unwrap_or_else(|e| {
            eprintln!("Invalid root_old: {e}");
            std::process::exit(2);
        });

    let trials: usize = std::env::args()
        .nth(3)
        .as_deref()
        .unwrap_or("7")
        .parse()
        .unwrap_or(7);

    if trials < 3 {
        eprintln!("Please use trials >= 3 (recommend odd, e.g., 7).");
        std::process::exit(2);
    }

    let g = load_reordered_csr_from_dir(&out_dir).unwrap_or_else(|e| {
        eprintln!("Load reordered CSR failed: {e}");
        std::process::exit(1);
    });
    validate_csr(&g).unwrap_or_else(|e| {
        eprintln!("CSR validation failed: {e}");
        std::process::exit(1);
    });

    let n = g.n() as u32;
    let m = g.m() as u64;

    let old2new_path = out_dir.join("cggraphRV1_5_old2new_u32.bin");
    let old2new = read_u32_bin(&old2new_path).unwrap_or_else(|e| {
        eprintln!("Read old2new failed: {e}");
        std::process::exit(1);
    });
    let root_new = old2new[root_old as usize];

    println!("Graph: n={n}, m={m}");
    println!("root_old={root_old} -> root_new={root_new}");
    println!("trials={trials}");

    // warm-up (not measured)
    let warm = bfs(&g, root_new);
    std::hint::black_box(&warm);

    let mut cpu_times: Vec<f64> = Vec::with_capacity(trials);
    let mut cpu_reached_last = 0u64;
    let mut cpu_maxd_last = -1i32;

    for _ in 0..trials {
        let t = Instant::now();
        let dist_cpu = bfs(&g, root_new);
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        let (reached, maxd) = bfs_stats(&dist_cpu);
        cpu_reached_last = reached;
        cpu_maxd_last = maxd;
        cpu_times.push(ms);
    }

    let cpu_med = median_ms(cpu_times.clone());
    cpu_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!(
        "[CPU BFS] min/median/max = {:.3} / {:.3} / {:.3} ms | reached={} maxd={}",
        cpu_times[0],
        cpu_med,
        cpu_times[cpu_times.len() - 1],
        cpu_reached_last,
        cpu_maxd_last
    );

    // gpu
    let _ctx = cust::quick_init()?;
    let module = Module::from_ptx(FRONTIER_EXPAND_PTX, &[])?;
    let kernel = module.get_function("frontier_write_neighbors_atomic_mark_dist_i32")?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    let d_offsets = DeviceBuffer::from_slice(&g.offsets)?;
    let d_dst = DeviceBuffer::from_slice(&g.dst)?;

    // Full-length init buffers (host) to reset each trial
    let mut visited_init: Vec<u32> = vec![0; n as usize];
    visited_init[root_new as usize] = 1;

    let mut dist_init: Vec<i32> = vec![-1; n as usize];
    dist_init[root_new as usize] = 0;

    // frontier_a init as full-length (cust copy requires equal lengths)
    let max_frontier = n as usize;
    let mut frontier_a_init: Vec<u32> = vec![0; max_frontier];
    frontier_a_init[0] = root_new;

    // Device buffers
    let mut d_visited = DeviceBuffer::<u32>::zeroed(n as usize)?;
    let mut d_dist = DeviceBuffer::<i32>::zeroed(n as usize)?;
    let mut d_frontier_a = DeviceBuffer::<u32>::zeroed(max_frontier)?;
    let mut d_frontier_b = DeviceBuffer::<u32>::zeroed(max_frontier)?;
    let mut d_write_count = DeviceBuffer::from_slice(&[0u32])?;

    // gpu warm-up (not measured)
    let (warm_ms, warm_reached, warm_maxd) = run_gpu_bfs_trial(
        &stream,
        &kernel,
        &d_offsets,
        &d_dst,
        &mut d_visited,
        &mut d_dist,
        &mut d_frontier_a,
        &mut d_frontier_b,
        &mut d_write_count,
        &visited_init,
        &dist_init,
        &frontier_a_init,
        root_new,
        n,
    )?;
    std::hint::black_box((warm_ms, warm_reached, warm_maxd));

    // gpu trials
    let mut gpu_times: Vec<f64> = Vec::with_capacity(trials);
    let mut gpu_reached_last = 0u64;
    let mut gpu_maxd_last = -1i32;

    for _ in 0..trials {
        let (ms, reached, maxd) = run_gpu_bfs_trial(
            &stream,
            &kernel,
            &d_offsets,
            &d_dst,
            &mut d_visited,
            &mut d_dist,
            &mut d_frontier_a,
            &mut d_frontier_b,
            &mut d_write_count,
            &visited_init,
            &dist_init,
            &frontier_a_init,
            root_new,
            n,
        )?;
        gpu_reached_last = reached;
        gpu_maxd_last = maxd;
        gpu_times.push(ms);
    }

    let gpu_med = median_ms(gpu_times.clone());
    gpu_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!(
        "[GPU BFS] min/median/max = {:.3} / {:.3} / {:.3} ms | reached={} maxd={}",
        gpu_times[0],
        gpu_med,
        gpu_times[gpu_times.len() - 1],
        gpu_reached_last,
        gpu_maxd_last
    );

    //cross-check
    println!(
        "[Compare] reached GPU={} vs CPU={}, maxd GPU={} vs CPU={}",
        gpu_reached_last, cpu_reached_last, gpu_maxd_last, cpu_maxd_last
    );

    if gpu_reached_last != cpu_reached_last || gpu_maxd_last != cpu_maxd_last {
        eprintln!("[WARN] GPU and CPU stats differ; investigate correctness before reporting speedup.");
    }

    Ok(())
}
