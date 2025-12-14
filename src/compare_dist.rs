use std::error::Error;
use std::path::Path;

use rust_het_graph::graph::io_bin::{read_i32_bin, read_u64_bin};

pub fn compare_bfs_i32(rust_path: &Path, cg_path: &Path) -> Result<(), Box<dyn Error>> {
    let r = read_i32_bin(rust_path)?;
    let c = read_i32_bin(cg_path)?;

    if r.len() != c.len() {
        return Err(format!("BFS len mismatch: rust={} cg={}", r.len(), c.len()).into());
    }

    let mut mismatches = 0usize;
    for i in 0..r.len() {
        if r[i] != c[i] {
            mismatches += 1;
            if mismatches <= 10 {
                eprintln!("[BFS mismatch] idx={i} rust={} cg={}", r[i], c[i]);
            }
        }
    }

    if mismatches == 0 {
        println!("[BFS compare] OK: all {} entries match", r.len());
    } else {
        println!("[BFS compare] mismatches = {mismatches} / {}", r.len());
    }
    Ok(())
}

pub fn compare_sssp_u64(rust_path: &Path, cg_path: &Path) -> Result<(), Box<dyn Error>> {
    let r = read_u64_bin(rust_path)?;
    let c = read_u64_bin(cg_path)?;

    if r.len() != c.len() {
        return Err(format!("SSSP len mismatch: rust={} cg={}", r.len(), c.len()).into());
    }

    let mut mismatches = 0usize;
    for i in 0..r.len() {
        if r[i] != c[i] {
            mismatches += 1;
            if mismatches <= 10 {
                eprintln!("[SSSP mismatch] idx={i} rust={} cg={}", r[i], c[i]);
            }
        }
    }

    if mismatches == 0 {
        println!("[SSSP compare] OK: all {} entries match", r.len());
    } else {
        println!("[SSSP compare] mismatches = {mismatches} / {}", r.len());
    }
    Ok(())
}
