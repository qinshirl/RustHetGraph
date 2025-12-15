#!/usr/bin/env python3
"""
plot_compare_bfs_logs.py

Given a parsed directory created by parse_compare_bfs_logs.py, generate:
- <policy>_trials.png
- <policy>_frontier_by_level.png (if levels CSV exists)
- <policy>_gpu_fraction_by_level.png (if levels CSV exists)
- median_comparison.png (all policies)

Example:
  python3 plot_compare_bfs_logs.py --parsed results_log/parsed --out results_log/plots
"""
import argparse, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parsed", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    summary = pd.read_csv(os.path.join(args.parsed, "perf_summary.csv"))

    # Per-policy plots
    for _, row in summary.iterrows():
        policy = row["policy"]

        trials_csv = os.path.join(args.parsed, f"{policy}_trials.csv")
        if os.path.exists(trials_csv):
            df = pd.read_csv(trials_csv)
            plt.figure(figsize=(8, 4))
            plt.plot(df["trial"], df["time_ms"], marker="o")
            plt.xlabel("Trial")
            plt.ylabel("Time (ms)")
            plt.title(f"RustHetGraph coop BFS trial times ({policy})")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(args.out, f"{policy}_trials.png"), dpi=200)
            plt.close()

        levels_csv = os.path.join(args.parsed, f"{policy}_levels.csv")
        if os.path.exists(levels_csv):
            lv = pd.read_csv(levels_csv).sort_values("lvl")
            if "gpu_frac" not in lv.columns:
                lv["gpu_frac"] = lv["gpu_frontier"] / lv["frontier"].replace(0, np.nan)

            plt.figure(figsize=(8, 4))
            plt.plot(lv["lvl"], lv["frontier"], marker="o", label="frontier")
            plt.plot(lv["lvl"], lv["gpu_frontier"], marker="o", label="gpu_frontier")
            plt.plot(lv["lvl"], lv["cpu_frontier"], marker="o", label="cpu_frontier")
            plt.xlabel("BFS Level")
            plt.ylabel("Vertices in frontier")
            plt.title(f"Frontier split by level ({policy})")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(args.out, f"{policy}_frontier_by_level.png"), dpi=200)
            plt.close()

            plt.figure(figsize=(8, 4))
            plt.plot(lv["lvl"], lv["gpu_frac"], marker="o")
            plt.xlabel("BFS Level")
            plt.ylabel("GPU share of frontier")
            plt.title(f"GPU fraction of frontier by level ({policy})")
            plt.ylim(0, 1.05)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(args.out, f"{policy}_gpu_fraction_by_level.png"), dpi=200)
            plt.close()

    # Overall median comparison (Rust vs CG CPU vs CG GPU)
    plt.figure(figsize=(8, 4))
    x = np.arange(len(summary["policy"]))
    w = 0.25
    plt.bar(x - w, summary["rust_coop_median_ms"], width=w, label="RustHetGraph (coop median)")
    plt.bar(x, summary["cg_cpu_median_ms"], width=w, label="CGgraph CPU (median)")
    plt.bar(x + w, summary["cg_gpu_median_ms"], width=w, label="CGgraph GPU (median)")
    plt.xticks(x, summary["policy"])
    plt.ylabel("Time (ms)")
    plt.title("Median BFS runtime comparison")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "median_comparison.png"), dpi=200)
    plt.close()

    print(f"Wrote plots to: {args.out}")

if __name__ == "__main__":
    main()
