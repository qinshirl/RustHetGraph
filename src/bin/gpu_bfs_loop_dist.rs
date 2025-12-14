use cust::prelude::*;
use rust_het_graph::gpu::runtime::FRONTIER_EXPAND_PTX;

use rust_het_graph::alg::bfs::bfs;
use rust_het_graph::graph::io_bin::{load_reordered_csr_from_dir, read_u32_bin};
use rust_het_graph::graph::stats::validate_csr;

use std::path::PathBuf;
use std::time::Instant;

fn bfs_stats(dist: &[i32]) -> (u64, i32) {
    let mut reached: u64 = 0;
    let mut maxd: i32 = -1;
    for &d in dist {
        if d >= 0 {
            reached += 1;
            if d > maxd { maxd = d; }
        }
    }
    (reached, maxd)
}

fn main() -> cust::error::CudaResult<()> {
    // cargo run --release --bin gpu_bfs_loop_dist -- <bin_dir> [root_old]
    let out_dir = std::env::args().nth(1).map(PathBuf::from).unwrap_or_else(|| {
        eprintln!("Usage: cargo run --release --bin gpu_bfs_loop_dist -- <bin_dir> [root_old]");
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
    if old2new.len() != g.n() {
        eprintln!("old2new len {} != n {}", old2new.len(), g.n());
        std::process::exit(1);
    }
    let root_new = old2new[root_old as usize];

    println!("Graph: n={n}, m={m}");
    println!("root_old={root_old} -> root_new={root_new}");

    let t_cpu = Instant::now();
    let dist_cpu = bfs(&g, root_new);
    let cpu_ms = t_cpu.elapsed().as_secs_f64() * 1000.0;
    let (cpu_reached, cpu_maxd) = bfs_stats(&dist_cpu);
    println!(
        "[CPU BFS] time={:.3} ms, reached={}, maxd={}",
        cpu_ms, cpu_reached, cpu_maxd
    );

    let _ctx = cust::quick_init()?;
    let module = Module::from_ptx(FRONTIER_EXPAND_PTX, &[])?;
    let kernel = module.get_function("frontier_write_neighbors_atomic_mark_dist_i32")?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    let d_offsets = DeviceBuffer::from_slice(&g.offsets)?;
    let d_dst = DeviceBuffer::from_slice(&g.dst)?;

    let mut visited_host: Vec<u32> = vec![0; n as usize];
    visited_host[root_new as usize] = 1;
    let d_visited = DeviceBuffer::from_slice(&visited_host)?;

    let mut dist_init: Vec<i32> = vec![-1; n as usize];
    dist_init[root_new as usize] = 0;
    let d_dist = DeviceBuffer::from_slice(&dist_init)?;

    let max_frontier = n as usize;

    let mut frontier_a_host = vec![0u32; max_frontier];
    frontier_a_host[0] = root_new;
    let mut d_frontier_a = DeviceBuffer::from_slice(&frontier_a_host)?;
    let mut d_frontier_b = DeviceBuffer::<u32>::zeroed(max_frontier)?;
    let mut d_write_count = DeviceBuffer::from_slice(&[0u32])?;

    let mut frontier_len: u32 = 1;
    let mut level: i32 = 0;

    let threads: u32 = 256;

    let t_gpu_total = Instant::now();
    while frontier_len > 0 {
        d_write_count.copy_from(&[0u32])?;

        let blocks: u32 = (frontier_len + threads - 1) / threads;

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

        let mut h_count = [0u32];
        d_write_count.copy_to(&mut h_count)?;
        let next_len = h_count[0];

        println!(
            "[GPU BFS] level={} frontier_len={} next_len={}",
            level, frontier_len, next_len
        );

        std::mem::swap(&mut d_frontier_a, &mut d_frontier_b);

        frontier_len = next_len;
        level += 1;

        if level > n as i32 {
            panic!("BFS level exceeded n; something is wrong.");
        }
    }
    let gpu_ms_total = t_gpu_total.elapsed().as_secs_f64() * 1000.0;


    let mut dist_gpu: Vec<i32> = vec![0; n as usize];
    d_dist.copy_to(&mut dist_gpu)?;
    let (gpu_reached, gpu_maxd) = bfs_stats(&dist_gpu);

    println!(
        "\n[GPU BFS] total_time={:.3} ms, reached={}, maxd={}",
        gpu_ms_total, gpu_reached, gpu_maxd
    );

    println!(
        "[Compare] reached: GPU={} vs CPU={}, maxd: GPU={} vs CPU={}",
        gpu_reached, cpu_reached, gpu_maxd, cpu_maxd
    );

    let sample_idx: [usize; 8] = [0, 1, 2, 3, 10, 100, 1000, (n as usize).saturating_sub(1)];
    for &idx in &sample_idx {
        if idx < dist_cpu.len() {
            if dist_gpu[idx] != dist_cpu[idx] {
                eprintln!(
                    "[WARN] dist mismatch at idx {}: gpu={}, cpu={}",
                    idx, dist_gpu[idx], dist_cpu[idx]
                );
            }
        }
    }

    Ok(())
}
