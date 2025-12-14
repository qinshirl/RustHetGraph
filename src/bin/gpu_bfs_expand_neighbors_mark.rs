use cust::prelude::*;
use rust_het_graph::gpu::runtime::FRONTIER_EXPAND_PTX;

use rust_het_graph::graph::io_bin::{load_reordered_csr_from_dir, read_u32_bin};
use rust_het_graph::graph::stats::validate_csr;

use std::path::PathBuf;
use std::time::Instant;

fn main() -> cust::error::CudaResult<()> {
    // cargo run --release --bin gpu_bfs_expand_neighbors_mark -- <bin_dir> [root_old]
    let out_dir = std::env::args().nth(1).map(PathBuf::from).unwrap_or_else(|| {
        eprintln!(
            "Usage: cargo run --release --bin gpu_bfs_expand_neighbors_mark -- <bin_dir> [root_old]"
        );
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

    let root_new = old2new[root_old as usize];

    println!("Graph: n={n}, m={m}");
    println!("root_old={root_old} -> root_new={root_new}");

    let s = g.offsets[root_new as usize] as usize;
    let e = g.offsets[root_new as usize + 1] as usize;
    let frontier: Vec<u32> = g.dst[s..e].to_vec();
    let frontier_len = frontier.len() as u32;

    println!("Frontier = neighbors(root_new): frontier_len={frontier_len}");

    if frontier_len == 0 {
        println!("Empty frontier, nothing to do.");
        return Ok(());
    }

    let mut visited_cpu: Vec<u32> = vec![0; n as usize];
    visited_cpu[root_new as usize] = 1;

    let mut expected_cpu: Vec<u32> = Vec::new();
    for &u in &frontier {
        let us = g.offsets[u as usize] as usize;
        let ue = g.offsets[u as usize + 1] as usize;
        for &v in &g.dst[us..ue] {
            if visited_cpu[v as usize] == 0 {
                visited_cpu[v as usize] = 1; // mark
                expected_cpu.push(v);
            }
        }
    }

    println!(
        "CPU reference produced {} unique next-frontier vertices",
        expected_cpu.len()
    );


    let _ctx = cust::quick_init()?;
    let module = Module::from_ptx(FRONTIER_EXPAND_PTX, &[])?;
    let kernel = module.get_function("frontier_write_neighbors_atomic_mark_u32")?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    let d_offsets = DeviceBuffer::from_slice(&g.offsets)?;
    let d_dst = DeviceBuffer::from_slice(&g.dst)?;
    let d_frontier = DeviceBuffer::from_slice(&frontier)?;

    let mut visited_gpu = vec![0u32; n as usize];
    visited_gpu[root_new as usize] = 1;
    let d_visited = DeviceBuffer::from_slice(&visited_gpu)?;

    // worst case output buffer = sum of degrees
    let total_out = frontier
        .iter()
        .map(|&u| {
            (g.offsets[u as usize + 1] - g.offsets[u as usize]) as usize
        })
        .sum::<usize>();

    let mut d_candidates = DeviceBuffer::<u32>::zeroed(total_out)?;
    let d_write_count = DeviceBuffer::from_slice(&[0u32])?;

    let threads: u32 = 256;
    let blocks: u32 = (frontier_len + threads - 1) / threads;

    let t0 = Instant::now();
    unsafe {
        launch!(
            kernel<<<blocks, threads, 0, stream>>>(
                d_offsets.as_device_ptr(),
                d_dst.as_device_ptr(),
                d_frontier.as_device_ptr(),
                d_visited.as_device_ptr(),
                d_candidates.as_device_ptr(),
                d_write_count.as_device_ptr(),
                frontier_len
            )
        )?;
    }
    stream.synchronize()?;
    let gpu_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // Read back write_count
    let mut h_count = [0u32];
    d_write_count.copy_to(&mut h_count)?;
    let packed_len = h_count[0] as usize;

    println!(
        "GPU kernel done: packed_len={}, time={:.3} ms",
        packed_len, gpu_ms
    );

    // Copy packed candidates
    let mut tmp = vec![0u32; total_out];
    d_candidates.copy_to(&mut tmp)?;
    let mut gpu_out = tmp[..packed_len].to_vec();

    if gpu_out.len() != expected_cpu.len() {
        panic!(
            "Length mismatch: gpu={}, cpu={}",
            gpu_out.len(),
            expected_cpu.len()
        );
    }

    gpu_out.sort_unstable();
    let mut cpu_sorted = expected_cpu.clone();
    cpu_sorted.sort_unstable();

    if gpu_out != cpu_sorted {
        panic!("Mismatch between GPU and CPU visited-mark expansion");
    }

    println!(
        "Validation OK: {} unique vertices discovered (no duplicates).",
        gpu_out.len()
    );

    Ok(())
}
