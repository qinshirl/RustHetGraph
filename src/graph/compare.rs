use std::cmp::Ordering;

use crate::graph::fps::SpeedRecord;

#[derive(Debug, Clone)]
pub struct AlgoSummary {
    pub rust_time_ms: f64,
    pub rust_reached: u64,
    pub rust_max_dist_label: String,
    pub cg_outer_runs: usize,
    pub cg_median_total_ms: f64,
    pub cg_min_total_ms: f64,
    pub cg_max_total_ms: f64,
    pub cg_median_steps: usize,
}

fn median_f64(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let n = xs.len();
    if n == 0 {
        return f64::NAN;
    }
    if n % 2 == 1 {
        xs[n / 2]
    } else {
        (xs[n / 2 - 1] + xs[n / 2]) / 2.0
    }
}

fn median_usize(mut xs: Vec<usize>) -> usize {
    xs.sort_unstable();
    let n = xs.len();
    if n == 0 {
        return 0;
    }
    if n % 2 == 1 {
        xs[n / 2]
    } else {
        xs[n / 2]
    }
}

/// Extract per-run total time from CGgraph records.
/// We treat "run total" as the last record's total_time_ms if present,
/// else we fall back to summing time_ms.
fn cg_run_total_ms(run: &[SpeedRecord]) -> f64 {
    if let Some(last) = run.last() {
        if last.total_time_ms > 0.0 {
            return last.total_time_ms;
        }
    }
    run.iter().map(|r| r.time_ms).sum()
}

/// Build a summary comparing Rust to CGgraph (CPU FPS).
pub fn summarize_vs_cggraph(
    algo_name: &str,
    rust_time_ms: f64,
    rust_reached: u64,
    rust_max_dist_label: String,
    cg_runs: &[Vec<SpeedRecord>],
) -> AlgoSummary {
    let cg_outer_runs = cg_runs.len();

    let mut totals: Vec<f64> = Vec::with_capacity(cg_outer_runs);
    let mut steps: Vec<usize> = Vec::with_capacity(cg_outer_runs);

    for run in cg_runs {
        totals.push(cg_run_total_ms(run));
        steps.push(run.len());
    }

    let cg_median_total_ms = median_f64(totals.clone());
    let (cg_min_total_ms, cg_max_total_ms) = if totals.is_empty() {
        (f64::NAN, f64::NAN)
    } else {
        let mut t = totals.clone();
        t.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        (t[0], t[t.len() - 1])
    };

    let cg_median_steps = median_usize(steps);

    let summary = AlgoSummary {
        rust_time_ms,
        rust_reached,
        rust_max_dist_label,
        cg_outer_runs,
        cg_median_total_ms,
        cg_min_total_ms,
        cg_max_total_ms,
        cg_median_steps,
    };

    println!("\n[D5] Summary: {algo_name}");
    println!("  Rust:   {:>10.3} ms | reached {:>8} | maxDist {}", summary.rust_time_ms, summary.rust_reached, summary.rust_max_dist_label);
    println!(
        "  CG CPU: {:>10.3} ms (median over {} runs; min {:.3}, max {:.3}) | median steps {}",
        summary.cg_median_total_ms,
        summary.cg_outer_runs,
        summary.cg_min_total_ms,
        summary.cg_max_total_ms,
        summary.cg_median_steps
    );

    if summary.cg_median_total_ms.is_finite() && summary.cg_median_total_ms > 0.0 {
        let speedup = summary.cg_median_total_ms / summary.rust_time_ms.max(1e-9);
        println!("  Ratio: CG_median / Rust = {:.3}x", speedup);
    } else {
        println!("  Ratio: n/a");
    }

    summary
}
