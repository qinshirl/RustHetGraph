use cust::prelude::*;
use rust_het_graph::gpu::runtime::FRONTIER_EXPAND_PTX;

use rust_het_graph::graph::io_bin::{load_reordered_csr_from_dir, read_u32_bin};
use rust_het_graph::graph::stats::validate_csr;

use std::path::PathBuf;
use std::time::Instant;

fn exclusive_scan_u32(input: &[u32]) -> (Vec<u32>, u32) {
    // out[i] = sum_{j < i} input[j], and return total sum
    let mut out = Vec::with_capacity(input.len());
    let mut acc: u32 = 0;
    for &x in input {
        out.push(acc);
        acc = acc.wrapping_add(x);
    }
    (out, acc)
}

fn main() -> cust::error::CudaResult<()> {
    // cargo run --release --bin gpu_bfs_expand_neighbors_atomic -- <bin_dir> [root_old]
    let out_dir = std::env::args().nth(1).map(PathBuf::from).unwrap_or_else(|| {
        eprintln!("Usage: cargo run --release --bin gpu_bfs_expand_neighbors_atomic -- <bin_dir> [root_old]");
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

    // Frontier = neighbors(root_new)
    let root_start = g.offsets[root_new as usize] as usize;
    let root_end = g.offsets[root_new as usize + 1] as usize;
    let frontier: Vec<u32> = g.dst[root_start..root_end].to_vec();
    let frontier_len = frontier.len() as u32;

    println!("Frontier = neighbors(root_new): frontier_len={frontier_len}");

    if frontier_len == 0 {
        println!("Frontier is empty; nothing to expand. Exiting.");
        return Ok(());
    }

    // visited: only root visited for now
    let mut visited_cpu: Vec<u8> = vec![0; n as usize];
    visited_cpu[root_new as usize] = 1;

    //CUDA init + module
    let _ctx = cust::quick_init()?;
    let module = Module::from_ptx(FRONTIER_EXPAND_PTX, &[])?;
    let f_deg = module.get_function("frontier_degrees_u32")?;
    let f_atomic = module.get_function("frontier_write_neighbors_atomic_u32")?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    // Upload CSR + frontier + visited
    let d_offsets = DeviceBuffer::from_slice(&g.offsets)?;
    let d_dst = DeviceBuffer::from_slice(&g.dst)?;
    let d_frontier = DeviceBuffer::from_slice(&frontier)?;
    let d_visited = DeviceBuffer::from_slice(&visited_cpu)?;

    // Pass 1 degrees (to size worst-case output buffer = total_out)
    let mut d_deg = DeviceBuffer::<u32>::zeroed(frontier_len as usize)?;

    let threads: u32 = 256;
    let blocks: u32 = (frontier_len + threads - 1) / threads;

    let t0 = Instant::now();
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

    let (_out_offsets_host, total_out) = exclusive_scan_u32(&deg_host);
    println!(
        "GPU pass1 done: time={:.3} ms, total_out(worst-case)={total_out}",
        pass1_ms
    );

    // Allocate candidates_out of size total_out (worst case)
    let mut d_candidates = DeviceBuffer::<u32>::zeroed(total_out as usize)?;

    // Device-side write counter initialized to 0
    let h_zero: [u32; 1] = [0];
    let mut d_write_count = DeviceBuffer::from_slice(&h_zero)?;

    // Pass 2: atomic compaction
    let t1 = Instant::now();
    unsafe {
        launch!(
            f_atomic<<<blocks, threads, 0, stream>>>(
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
    let pass2_ms = t1.elapsed().as_secs_f64() * 1000.0;

    // Read back write_count to know actual packed length
    let mut h_count: [u32; 1] = [0];
    d_write_count.copy_to(&mut h_count)?;
    let packed_len = h_count[0] as usize;

    println!(
        "GPU pass2 atomic done: packed_len={}, time={:.3} ms (buffer_cap={})",
        packed_len, pass2_ms, total_out
    );


    // let mut candidates_host: Vec<u32> = vec![0; packed_len];
    // if packed_len > 0 {
    //     d_candidates.copy_to(&mut candidates_host)?;
    //     candidates_host.truncate(packed_len);
    // }

    let mut candidates_host_full: Vec<u32> = vec![0; total_out as usize];
    d_candidates.copy_to(&mut candidates_host_full)?;
    let candidates_host = candidates_host_full[..packed_len].to_vec();


    let mut expected_cpu: Vec<u32> = Vec::new();
    expected_cpu.reserve(total_out as usize);

    for &u in &frontier {
        let s = g.offsets[u as usize] as usize;
        let e = g.offsets[u as usize + 1] as usize;
        for &v in &g.dst[s..e] {
            if visited_cpu[v as usize] == 0 {
                expected_cpu.push(v);
            }
        }
    }

    if candidates_host.len() != expected_cpu.len() {
        panic!(
            "Length mismatch: gpu_packed_len={}, cpu_expected_len={}",
            candidates_host.len(),
            expected_cpu.len()
        );
    }
    let mut gpu_sorted = candidates_host.clone();
    let mut cpu_sorted = expected_cpu.clone();
    gpu_sorted.sort_unstable();
    cpu_sorted.sort_unstable();

    if gpu_sorted != cpu_sorted {
        panic!("Content mismatch after sorting (multiset check failed).");
    }

    println!(
        "Validation OK (multiset). packed_len={} (duplicates allowed, no dedup yet).",
        packed_len
    );

    Ok(())
}
