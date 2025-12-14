use cust::prelude::*;
use rust_het_graph::gpu::runtime::FRONTIER_EXPAND_PTX;

use rust_het_graph::graph::io_bin::{load_reordered_csr_from_dir, read_u32_bin};
use rust_het_graph::graph::stats::validate_csr;

use std::path::PathBuf;
use std::time::Instant;

fn main() -> cust::error::CudaResult<()> {
    // Usage:
    // cargo run --release --bin gpu_bfs_expand1 -- <bin_dir> [root_old]
    let out_dir = std::env::args().nth(1).map(PathBuf::from).unwrap_or_else(|| {
        eprintln!("Usage: cargo run --release --bin gpu_bfs_expand1 -- <bin_dir> [root_old]");
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

    // Load reordered CSR
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

    // Load old2new mapping to map root_old -> root_new
    let old2new_path = out_dir.join("cggraphRV1_5_old2new_u32.bin");
    let old2new = read_u32_bin(&old2new_path).unwrap_or_else(|e| {
        eprintln!("Read old2new failed: {e}");
        std::process::exit(1);
    });

    if old2new.len() != g.n() {
        eprintln!("old2new len {} != n {}", old2new.len(), g.n());
        std::process::exit(1);
    }

    let root_new = *old2new.get(root_old as usize).unwrap_or_else(|| {
        eprintln!("root_old {root_old} out of range");
        std::process::exit(1);
    });

    println!("Graph: n={n}, m={m}");
    println!("root_old={root_old} -> root_new={root_new}");

    // CPU expected neighbors for single-vertex frontier
    let start = g.offsets[root_new as usize] as usize;
    let end = g.offsets[root_new as usize + 1] as usize;
    let expected = &g.dst[start..end];
    let expected_deg = (end - start) as u32;

    println!("CPU expected degree(root_new) = {expected_deg}");

    // CUDA init
    let _ctx = cust::quick_init()?;
    let module = Module::from_ptx(FRONTIER_EXPAND_PTX, &[])?;
    let f_deg = module.get_function("frontier_degrees_u32")?;
    let f_write = module.get_function("frontier_write_neighbors_u32")?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    // Frontier (single vertex)
    let frontier_len: u32 = 1;
    let h_frontier: [u32; 1] = [root_new];

    // Device buffers
    let d_offsets = DeviceBuffer::from_slice(&g.offsets)?;
    let d_dst = DeviceBuffer::from_slice(&g.dst)?;
    let d_frontier = DeviceBuffer::from_slice(&h_frontier)?;

    // Pass 1: degrees
    let mut d_deg = DeviceBuffer::<u32>::zeroed(frontier_len as usize)?;

    let t0 = Instant::now();
    unsafe {
        launch!(
            f_deg<<<1, 32, 0, stream>>>(
                d_offsets.as_device_ptr(),
                d_frontier.as_device_ptr(),
                d_deg.as_device_ptr(),
                frontier_len
            )
        )?;
    }
    stream.synchronize()?;
    let pass1_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let mut h_deg: [u32; 1] = [0];
    d_deg.copy_to(&mut h_deg)?;
    let deg_gpu = h_deg[0];

    println!("GPU pass1 degree = {deg_gpu} (time {:.3} ms)", pass1_ms);

    if deg_gpu != expected_deg {
        panic!("Degree mismatch: gpu={deg_gpu}, cpu={expected_deg}");
    }

    // CPU exclusive scan: for len=1, out_offsets[0]=0
    let h_out_offsets: [u32; 1] = [0];
    let total_out = deg_gpu as usize;

    // Pass 2: write neighbors
    let d_out_offsets = DeviceBuffer::from_slice(&h_out_offsets)?;
    let mut d_candidates = DeviceBuffer::<u32>::zeroed(total_out)?;

    let t1 = Instant::now();
    unsafe {
        launch!(
            f_write<<<1, 32, 0, stream>>>(
                d_offsets.as_device_ptr(),
                d_dst.as_device_ptr(),
                d_frontier.as_device_ptr(),
                d_out_offsets.as_device_ptr(),
                d_candidates.as_device_ptr(),
                frontier_len
            )
        )?;
    }
    stream.synchronize()?;
    let pass2_ms = t1.elapsed().as_secs_f64() * 1000.0;

    let mut h_candidates: Vec<u32> = vec![0; total_out];
    d_candidates.copy_to(&mut h_candidates)?;

    println!(
        "GPU pass2 wrote {} neighbors (time {:.3} ms)",
        total_out, pass2_ms
    );

    // Validate exact order against CPU slice
    if h_candidates.len() != expected.len() {
        panic!(
            "Neighbor list length mismatch: gpu={}, cpu={}",
            h_candidates.len(),
            expected.len()
        );
    }

    for (i, (&gpu_v, &cpu_v)) in h_candidates.iter().zip(expected.iter()).enumerate() {
        if gpu_v != cpu_v {
            panic!("Neighbor mismatch at idx={i}: gpu={gpu_v}, cpu={cpu_v}");
        }
    }

    println!("GPU single-vertex frontier expansion OK (exact match).");
    Ok(())
}
