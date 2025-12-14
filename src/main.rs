// mod graph;
mod compare_dist;


use std::error::Error;
use std::hint::black_box;


use std::path::PathBuf;
use std::time::Instant;

use rust_het_graph::graph::io_bin::{
    load_csr_from_dir,
    load_reordered_csr_from_dir,
    read_u32_bin,
    write_u32_bin,
    write_i32_bin,
    write_u64_bin,
};
use rust_het_graph::graph::stats::{validate_csr, degree_summary};
use rust_het_graph::graph::reorder::cggraph_rank_v15;
use rust_het_graph::graph::permute::permute_csr_cggraph_v15;

use rust_het_graph::alg::bfs::bfs;
use rust_het_graph::alg::sssp::dijkstra_sssp;
use rust_het_graph::graph::fps::{read_speed_records, print_speed_summary};
use rust_het_graph::graph::compare::summarize_vs_cggraph;


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
            if d > maxd { maxd = d; }
        }
    }
    (reached, maxd)
}

fn sssp_stats(dist: &[u64]) -> (u64, u64) {
    let mut reached: u64 = 0;
    let mut maxd: u64 = 0;
    for &d in dist {
        if d != u64::MAX {
            reached += 1;
            if d > maxd { maxd = d; }
        }
    }
    (reached, maxd)
}




fn main() -> Result<(), Box<dyn Error>> {

    // Usage:
    // cargo run --release -- /data/webgraph/bin/web-Google /data/webgraph/bin/web-Google
    //
    // arg1: input bin dir (must contain native_csrOffset_u32.bin etc)
    // arg2: output dir for rank/old2new (optional; defaults to input dir)
    let in_dir = std::env::args().nth(1).map(PathBuf::from)
        .unwrap_or_else(|| {
            eprintln!("Usage: cargo run --release -- <bin_dir> [out_dir]");
            std::process::exit(2);
        });

    let out_dir = std::env::args().nth(2).map(PathBuf::from)
        .unwrap_or_else(|| in_dir.clone());

    let g = load_csr_from_dir(&in_dir).unwrap_or_else(|e| {
        eprintln!("Load failed: {e}");
        std::process::exit(1);
    });

    validate_csr(&g).unwrap_or_else(|e| {
        eprintln!("CSR validation failed: {e}");
        std::process::exit(1);
    });

    let (min_d, max_d, avg_d) = degree_summary(&g);

    println!("Loaded CSR from {:?}", in_dir);
    println!("n (vertices): {}", g.n());
    println!("m (edges):    {}", g.m());
    println!("weights:      {}", if g.w.is_some() { "yes" } else { "no" });
    println!("out-degree min/max/avg: {min_d} / {max_d} / {avg_d:.3}");

    println!("\nRunning CGgraph v1.5 reorder (rank + old2new) ...");
    let t0 = Instant::now();
    let res = cggraph_rank_v15(&g).unwrap_or_else(|e| {
        eprintln!("Reorder failed: {e}");
        std::process::exit(1);
    });
    let dt = t0.elapsed();
    println!("Reorder done in {:.3}s", dt.as_secs_f64());
    println!("rank len: {}, old2new len: {}", res.rank.len(), res.old2new.len());

    // Write outputs (CGgraph-style artifacts)
    let rank_path = out_dir.join("cggraphRV1_5_rank_u32.bin");
    let old2new_path = out_dir.join("cggraphRV1_5_old2new_u32.bin");

    write_u32_bin(&rank_path, &res.rank).unwrap_or_else(|e| {
        eprintln!("Write rank failed: {e}");
        std::process::exit(1);
    });
    write_u32_bin(&old2new_path, &res.old2new).unwrap_or_else(|e| {
        eprintln!("Write old2new failed: {e}");
        std::process::exit(1);
    });

    println!("Wrote {:?}", rank_path);
    println!("Wrote {:?}", old2new_path);

    // ---------------- Milestone C ----------------
    println!("\nBuilding reordered CSR (CGgraph v1.5) ...");
    let t1 = Instant::now();
    let g2 = permute_csr_cggraph_v15(&g, &res.rank, &res.old2new).unwrap_or_else(|e| {
        eprintln!("Permute CSR failed: {e}");
        std::process::exit(1);
    });
    let dt2 = t1.elapsed();
    println!("Reordered CSR built in {:.3}s", dt2.as_secs_f64());

    // Write reordered CSR bins (CGgraph-style artifacts)
    let csr_off_path = out_dir.join("cggraphRV1_5_csrOffset_u32.bin");
    let csr_dst_path = out_dir.join("cggraphRV1_5_csrDest_u32.bin");
    let csr_w_path   = out_dir.join("cggraphRV1_5_csrWeight_u32.bin");

    write_u32_bin(&csr_off_path, &g2.offsets).unwrap_or_else(|e| {
        eprintln!("Write csrOffset failed: {e}");
        std::process::exit(1);
    });
    write_u32_bin(&csr_dst_path, &g2.dst).unwrap_or_else(|e| {
        eprintln!("Write csrDest failed: {e}");
        std::process::exit(1);
    });

    if let Some(w) = &g2.w {
        write_u32_bin(&csr_w_path, w).unwrap_or_else(|e| {
            eprintln!("Write csrWeight failed: {e}");
            std::process::exit(1);
        });
        println!("Wrote {:?}", csr_w_path);
    } else {
        println!("No weights present; skipped csrWeight output");
    }

    println!("Wrote {:?}", csr_off_path);
    println!("Wrote {:?}", csr_dst_path);

    // ---------------- Milestone D - Step D1 ----------------
    println!("\n[D1] Loading reordered CSR + old2new and mapping root ...");

    let g_re = load_reordered_csr_from_dir(&out_dir).unwrap_or_else(|e| {
        eprintln!("Load reordered CSR failed: {e}");
        std::process::exit(1);
    });

    validate_csr(&g_re).unwrap_or_else(|e| {
        eprintln!("Reordered CSR validation failed: {e}");
        std::process::exit(1);
    });

    // Load old2new mapping (produced in Milestone B)
    let old2new_path = out_dir.join("cggraphRV1_5_old2new_u32.bin");
    let old2new = read_u32_bin(&old2new_path).unwrap_or_else(|e| {
        eprintln!("Read old2new failed: {e}");
        std::process::exit(1);
    });

    if old2new.len() != g_re.n() {
        eprintln!(
            "old2new len {} does not match graph n {}",
            old2new.len(),
            g_re.n()
        );
        std::process::exit(1);
    }
    // Map root from old ID to new ID
    let root_old: u32 = std::env::args()
        .nth(3)
        .as_deref()
        .unwrap_or("0")
        .parse()
        .map_err(|e| format!("Invalid root_old arg (3rd): {e}"))?;

    let root_new = *old2new
        .get(root_old as usize)
        .ok_or_else(|| format!("root_old {root_old} out of range for n={}", g_re.n()))?;


    println!("[D1] root_old = {root_old}, root_new = {root_new}");
    println!("[D1] reordered CSR: n={}, m={}, weights={}",
        g_re.n(),
        g_re.m(),
        if g_re.w.is_some() { "yes" } else { "no" }
    );

    // ---------------- Milestone D - Step D2/D3 (Median timing) ----------------
    let trials: usize = 7; // choose odd number for clean median, e.g., 5 or 7

    println!("\n[D2] Running CPU BFS on reordered CSR (warm-up + median of {trials}) ...");
    // warm-up (not measured)
    let warm_bfs = bfs(&g_re, root_new);
    black_box(&warm_bfs);

    // measured runs
    let mut bfs_times_ms: Vec<f64> = Vec::with_capacity(trials);
    let mut dist_bfs: Vec<i32> = Vec::new();
    for _ in 0..trials {
        let t = Instant::now();
        let d = bfs(&g_re, root_new);
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        bfs_times_ms.push(ms);
        dist_bfs = d; // keep last run's output for stats + file write
    }
    let rust_bfs_ms = median_ms(bfs_times_ms);

    let mut bfs_times_sorted = bfs_times_ms.clone();
    bfs_times_sorted.sort_by(|a,b| a.partial_cmp(b).unwrap());
    println!("[D2] BFS times ms (sorted): {:?}", bfs_times_sorted);
    println!(
        "[D2] BFS min/median/max = {:.3} / {:.3} / {:.3} ms",
        bfs_times_sorted[0],
        rust_bfs_ms,
        bfs_times_sorted[bfs_times_sorted.len()-1]
    );

    let (bfs_reached, bfs_maxd) = bfs_stats(&dist_bfs);
    println!("[D2] BFS median time = {:.3} ms", rust_bfs_ms);
    println!("[D2] BFS reached {bfs_reached} / {} vertices, max distance = {bfs_maxd}", g_re.n());

    // write BFS output (new-id order)
    let bfs_path = out_dir.join("rust_bfs_dist_i32.bin");
    write_i32_bin(&bfs_path, &dist_bfs).unwrap_or_else(|e| {
        eprintln!("Write BFS dist failed: {e}");
        std::process::exit(1);
    });
    println!("[D2] Wrote {:?}", bfs_path);

    println!("\n[D3] Running CPU SSSP (warm-up + median of {trials}) ...");
    // warm-up (not measured)
    let warm_sssp = dijkstra_sssp(&g_re, root_new).unwrap_or_else(|e| {
        eprintln!("SSSP warm-up failed: {e}");
        std::process::exit(1);
    });
    black_box(&warm_sssp);

    // measured runs
    let mut sssp_times_ms: Vec<f64> = Vec::with_capacity(trials);
    let mut dist_sssp: Vec<u64> = Vec::new();
    for _ in 0..trials {
        let t = Instant::now();
        let d = dijkstra_sssp(&g_re, root_new).unwrap_or_else(|e| {
            eprintln!("SSSP failed: {e}");
            std::process::exit(1);
        });
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        sssp_times_ms.push(ms);
        dist_sssp = d; // keep last run for stats + file write
    }
    let rust_sssp_ms = median_ms(sssp_times_ms);
    let (sssp_reached, sssp_maxd) = sssp_stats(&dist_sssp);
    println!("[D3] SSSP median time = {:.3} ms", rust_sssp_ms);
    println!("[D3] SSSP reached {sssp_reached} / {} vertices, max distance = {sssp_maxd}", g_re.n());

    // write SSSP output
    let sssp_path = out_dir.join("rust_sssp_dist_u64.bin");
    write_u64_bin(&sssp_path, &dist_sssp).unwrap_or_else(|e| {
        eprintln!("Write SSSP dist failed: {e}");
        std::process::exit(1);
    });
    println!("[D3] Wrote {:?}", sssp_path);

    // ---------------- Milestone D - Step D4 (CGgraph FPS files) ----------------

    println!("\n[D4] Inspecting CGgraph FPS files (CPU) ...");

    let cg_bfs_path = "/home/qinshirl/ECE1724_rust/CGgraph_wsl/build/FPS_CPU_web-Google_CGgraphRV1_5_BFS.bin";
    let cg_sssp_path = "/home/qinshirl/ECE1724_rust/CGgraph_wsl/build/FPS_CPU_web-Google_CGgraphRV1_5_SSSP.bin";

    let bfs_runs = read_speed_records(cg_bfs_path)
        .map_err(|e| format!("read_speed_records BFS: {e}"))?;
    let sssp_runs = read_speed_records(cg_sssp_path)
        .map_err(|e| format!("read_speed_records SSSP: {e}"))?;
    
        println!("\n[D4] CG sanity check: sum(time_ms) vs stored_total_ms");
    for (i, run) in bfs_runs.iter().enumerate() {
        let (sum_ms, stored_ms) = rust_het_graph::graph::fps::run_time_sanity(run);
        println!("  CG BFS run {i}: sum(time_ms)={sum_ms:.3} ms | stored_total={stored_ms:.3} ms | diff={:.3} ms",
                stored_ms - sum_ms);
    }
    for (i, run) in sssp_runs.iter().enumerate() {
        let (sum_ms, stored_ms) = rust_het_graph::graph::fps::run_time_sanity(run);
        println!("  CG SSSP run {i}: sum(time_ms)={sum_ms:.3} ms | stored_total={stored_ms:.3} ms | diff={:.3} ms",
                stored_ms - sum_ms);
    }


    print_speed_summary("CG BFS", &bfs_runs, 20);
    print_speed_summary("CG SSSP", &sssp_runs, 20);

    summarize_vs_cggraph(
        "BFS",
        rust_bfs_ms,
        bfs_reached,
        format!("{bfs_maxd}"),
        &bfs_runs,
    );

    summarize_vs_cggraph(
        "SSSP",
        rust_sssp_ms,
        sssp_reached,
        format!("{sssp_maxd}"),
        &sssp_runs,
    );



    Ok(())

}
