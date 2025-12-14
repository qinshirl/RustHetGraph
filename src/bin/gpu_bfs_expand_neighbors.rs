use cust::prelude::*;
use rust_het_graph::gpu::runtime::FRONTIER_EXPAND_PTX;

use rust_het_graph::graph::io_bin::{load_reordered_csr_from_dir, read_u32_bin};
use rust_het_graph::graph::stats::validate_csr;

use std::path::PathBuf;
use std::time::Instant;

fn exclusive_scan_u32(input: &[u32]) -> (Vec<u32>, u32) {
    // out[i] = sum_{j < i} input[j]
    let mut out = Vec::with_capacity(input.len());
    let mut acc: u32 = 0;
    for &x in input {
        out.push(acc);
        acc = acc.wrapping_add(x);
    }
    (out, acc)
}

fn main() -> cust::error::CudaResult<()> {
    // Usage:
    // cargo run --release --bin gpu_bfs_expand_neighbors -- <bin_dir> [root_old]
    let out_dir = std::env::args().nth(1).map(PathBuf::from).unwrap_or_else(|| {
        eprintln!(
            "Usage: cargo run --release --bin gpu_bfs_expand_neighbors -- <bin_dir> [root_old]"
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

    // Map root_old -> root_new
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

    // Build frontier = neighbors(root_new) in CSR order
    let root_start = g.offsets[root_new as usize] as usize;
    let root_end = g.offsets[root_new as usize + 1] as usize;
    let frontier: Vec<u32> = g.dst[root_start..root_end].to_vec();
    let frontier_len = frontier.len() as u32;

    println!("Frontier = neighbors(root_new): frontier_len={frontier_len}");

    if frontier_len == 0 {
        println!("Frontier is empty; nothing to expand. Exiting.");
        return Ok(());
    }

    // CPU reference: for each frontier vertex u_i, the expected neighbor block is
    // dst[offsets[u_i]..offsets[u_i+1]] in CSR order.

    // ---------------- CUDA init + module ----------------
    let _ctx = cust::quick_init()?;
    let module = Module::from_ptx(FRONTIER_EXPAND_PTX, &[])?;
    let f_deg = module.get_function("frontier_degrees_u32")?;
    let f_write = module.get_function("frontier_write_neighbors_u32")?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    // Upload CSR buffers + frontier
    let d_offsets = DeviceBuffer::from_slice(&g.offsets)?;
    let d_dst = DeviceBuffer::from_slice(&g.dst)?;
    let d_frontier = DeviceBuffer::from_slice(&frontier)?;

    // Pass 1: degree per frontier vertex
    let mut d_deg = DeviceBuffer::<u32>::zeroed(frontier_len as usize)?;

    let t0 = Instant::now();
    let threads: u32 = 256;
    let blocks: u32 = (frontier_len + threads - 1) / threads;

    unsafe {
        launch!(
            f_deg<<<blocks, threads, 0, stream>>>(
                d_offsets.as_device_ptr(),
                d_frontier.as_device_ptr(),
                d_deg.as_device_ptr(),
                frontier_len
            )
        )?;
    }
    stream.synchronize()?;
    let pass1_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let mut deg_host: Vec<u32> = vec![0; frontier_len as usize];
    d_deg.copy_to(&mut deg_host)?;

    let (out_offsets_host, total_out) = exclusive_scan_u32(&deg_host);

    println!(
        "GPU pass1 done: blocks={blocks}, threads={threads}, time={:.3} ms, total_out={total_out}",
        pass1_ms
    );

    // Upload out_offsets and allocate candidates
    let d_out_offsets = DeviceBuffer::from_slice(&out_offsets_host)?;
    let mut d_candidates = DeviceBuffer::<u32>::zeroed(total_out as usize)?;

    // Pass 2: write neighbors packed by out_offsets
    let t1 = Instant::now();
    unsafe {
        launch!(
            f_write<<<blocks, threads, 0, stream>>>(
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

    let mut candidates_host: Vec<u32> = vec![0; total_out as usize];
    d_candidates.copy_to(&mut candidates_host)?;

    println!(
        "GPU pass2 done: wrote {} candidates, time={:.3} ms",
        total_out, pass2_ms
    );

    // ---------------- Validation (exact per-vertex block match) ----------------
    for i in 0..(frontier_len as usize) {
        let u = frontier[i];
        let deg = deg_host[i] as usize;
        let out_base = out_offsets_host[i] as usize;

        let cpu_start = g.offsets[u as usize] as usize;
        let cpu_end = g.offsets[u as usize + 1] as usize;
        let cpu_slice = &g.dst[cpu_start..cpu_end];

        if cpu_slice.len() != deg {
            panic!(
                "Degree mismatch at i={i}, u={u}: deg_host={}, cpu_deg={}",
                deg,
                cpu_slice.len()
            );
        }

        let gpu_slice = &candidates_host[out_base..out_base + deg];

        for (j, (&gv, &cv)) in gpu_slice.iter().zip(cpu_slice.iter()).enumerate() {
            if gv != cv {
                panic!(
                    "Mismatch at frontier_idx={i}, u={u}, local_idx={j}: gpu={gv}, cpu={cv}"
                );
            }
        }
    }

    println!(
        "GPU frontier expansion OK for frontier=neighbors(root_new). frontier_len={}, total_out={}",
        frontier_len, total_out
    );

    Ok(())
}
