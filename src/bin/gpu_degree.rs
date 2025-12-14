use cust::prelude::*;
use rust_het_graph::gpu::runtime::DEGREE_PTX;

use rust_het_graph::graph::io_bin::load_reordered_csr_from_dir;
use rust_het_graph::graph::stats::validate_csr;

use std::path::PathBuf;

fn main() -> cust::error::CudaResult<()> {
    let out_dir = std::env::args().nth(1).map(PathBuf::from).unwrap_or_else(|| {
        eprintln!("Usage: cargo run --release --bin gpu_degree -- <bin_dir>");
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

    let mut deg_cpu: Vec<u32> = vec![0; n as usize];
    for v in 0..(n as usize) {
        deg_cpu[v] = g.offsets[v + 1] - g.offsets[v];
    }

    let _ctx = cust::quick_init()?;
    let module = Module::from_ptx(DEGREE_PTX, &[])?;
    let func = module.get_function("csr_degree_u32")?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    let d_offsets = DeviceBuffer::from_slice(&g.offsets)?;
    let mut d_deg = DeviceBuffer::<u32>::zeroed(n as usize)?;

    let threads_per_block: u32 = 256;
    let blocks: u32 = (n + threads_per_block - 1) / threads_per_block;

    unsafe {
        launch!(
            func<<<blocks, threads_per_block, 0, stream>>>(
                d_offsets.as_device_ptr(),
                d_deg.as_device_ptr(),
                n
            )
        )?;
    }

    stream.synchronize()?;

    let mut deg_gpu: Vec<u32> = vec![0; n as usize];
    d_deg.copy_to(&mut deg_gpu)?;

    for i in 0..(n as usize) {
        if deg_gpu[i] != deg_cpu[i] {
            panic!(
                "Degree mismatch at v={i}: gpu={}, cpu={}",
                deg_gpu[i], deg_cpu[i]
            );
        }
    }

    let max_deg = deg_gpu.iter().copied().max().unwrap_or(0);
    let sum_deg: u64 = deg_gpu.iter().map(|&x| x as u64).sum();
    println!(
        "GPU degree kernel OK: n={}, max_deg={}, sum_deg={}",
        n, max_deg, sum_deg
    );

    Ok(())
}

