#!/usr/bin/env python3

import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


RE_POLICY = re.compile(r"\[INPUT\]\s+policy\s+=\s+([A-Za-z0-9_]+)")
RE_GRAPH = re.compile(r"\[GRAPH\]\s+n=(\d+),\s+m=(\d+)")
RE_TRIAL = re.compile(r"\[RUN\]\s+trial\s+(\d+):\s+([0-9.]+)\s+ms\s+\(reached=(\d+),\s+maxd=([-0-9]+)\)")
RE_RESULT_MIN_MED_MAX = re.compile(
    r"\[RESULT\]\s+coop BFS min/median/max\s+=\s+([0-9.]+)\s+/\s+([0-9.]+)\s+/\s+([0-9.]+)\s+ms"
)
RE_SUBGRAPH = re.compile(r"\[RESULT\]\s+subgraph:\s+policy=([A-Za-z0-9_]+),\s+m_gpu=(\d+),\s+edges_on_gpu=([0-9.]+)")
RE_LEVEL = re.compile(
    r"\[DEBUG\]\s+lvl=(\d+)\s+frontier=(\d+)\s+gpu_frontier=(\d+)\s+cpu_frontier=(\d+)\s+next=(\d+)\s+gpu_enabled=(\w+)"
)

RE_CG_CPU_SECTION = re.compile(r"\[D5\]\s+Summary:\s+BFS vs CG CPU([\s\S]*?)(?:\n\n|\Z)")
RE_CG_GPU_SECTION = re.compile(r"\[D5\]\s+Summary:\s+BFS vs CG GPU([\s\S]*?)(?:\n\n|\Z)")
RE_CG_MEDIAN = re.compile(
    r"CG CPU:\s+([0-9.]+)\s+ms\s+\(median over (\d+) runs;\s+min\s+([0-9.]+),\s+max\s+([0-9.]+)\)"
)


def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", s)


@dataclass
class ParsedLog:
    file: str
    policy: str
    n: int | None
    m: int | None
    m_gpu: int | None
    edges_on_gpu: float | None

    trials: list[dict]
    levels: list[dict]

    rust_min_ms: float | None
    rust_median_ms: float | None
    rust_max_ms: float | None

    reached_med: int | None
    maxd_med: int | None
    levels_count: int | None

    cg_cpu_median_ms: float | None
    cg_gpu_median_ms: float | None


def parse_log(path: str) -> ParsedLog:
    text = Path(path).read_text(errors="ignore")
    fname = os.path.basename(path)

    m = RE_POLICY.search(text)
    policy = m.group(1) if m else Path(path).stem

    m = RE_GRAPH.search(text)
    n = int(m.group(1)) if m else None
    m_edges = int(m.group(2)) if m else None

    trials = []
    for mm in RE_TRIAL.finditer(text):
        trials.append({
            "trial": int(mm.group(1)),
            "time_ms": float(mm.group(2)),
            "reached": int(mm.group(3)),
            "maxd": int(mm.group(4)),
        })

    rust_min = rust_median = rust_max = None
    mm = RE_RESULT_MIN_MED_MAX.search(text)
    if mm:
        rust_min = float(mm.group(1))
        rust_median = float(mm.group(2))
        rust_max = float(mm.group(3))

    m_gpu = edges_on_gpu = None
    mm = RE_SUBGRAPH.search(text)
    if mm:
        m_gpu = int(mm.group(2))
        edges_on_gpu = float(mm.group(3))

    seen = set()
    levels = []
    for mm in RE_LEVEL.finditer(text):
        lvl = int(mm.group(1))
        if lvl in seen:
            continue
        seen.add(lvl)
        levels.append({
            "lvl": lvl,
            "frontier": int(mm.group(2)),
            "gpu_frontier": int(mm.group(3)),
            "cpu_frontier": int(mm.group(4)),
            "next": int(mm.group(5)),
        })
    levels.sort(key=lambda d: d["lvl"])

    reached_med = int(np.median([t["reached"] for t in trials])) if trials else None
    maxd_med = int(np.median([t["maxd"] for t in trials])) if trials else None
    levels_count = (levels[-1]["lvl"] + 1) if levels else None

    cg_cpu_median = None
    sec = RE_CG_CPU_SECTION.search(text)
    if sec:
        mm = RE_CG_MEDIAN.search(sec.group(1))
        if mm:
            cg_cpu_median = float(mm.group(1))

    cg_gpu_median = None
    sec = RE_CG_GPU_SECTION.search(text)
    if sec:
        mm = RE_CG_MEDIAN.search(sec.group(1))
        if mm:
            cg_gpu_median = float(mm.group(1))

    return ParsedLog(
        file=fname,
        policy=policy,
        n=n,
        m=m_edges,
        m_gpu=m_gpu,
        edges_on_gpu=edges_on_gpu,
        trials=trials,
        levels=levels,
        rust_min_ms=rust_min,
        rust_median_ms=rust_median,
        rust_max_ms=rust_max,
        reached_med=reached_med,
        maxd_med=maxd_med,
        levels_count=levels_count,
        cg_cpu_median_ms=cg_cpu_median,
        cg_gpu_median_ms=cg_gpu_median,
    )


def plot_trials(p: ParsedLog, out_dir: str) -> str | None:
    if not p.trials:
        return None
    xs = [t["trial"] for t in p.trials]
    ys = [t["time_ms"] for t in p.trials]

    plt.figure(figsize=(8, 4))
    plt.plot(xs, ys, marker="o")
    plt.xlabel("Trial")
    plt.ylabel("Time (ms)")
    plt.title(f"RustHetGraph coop BFS trial times ({p.policy})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(out_dir, f"{safe_name(p.policy)}_trials.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def plot_frontier(p: ParsedLog, out_dir: str) -> tuple[str | None, str | None]:
    if not p.levels:
        return (None, None)

    lvl = [d["lvl"] for d in p.levels]
    frontier = [d["frontier"] for d in p.levels]
    gpu_frontier = [d["gpu_frontier"] for d in p.levels]
    cpu_frontier = [d["cpu_frontier"] for d in p.levels]

    plt.figure(figsize=(8, 4))
    plt.plot(lvl, frontier, marker="o", label="frontier")
    plt.plot(lvl, gpu_frontier, marker="o", label="gpu_frontier")
    plt.plot(lvl, cpu_frontier, marker="o", label="cpu_frontier")
    plt.xlabel("BFS Level")
    plt.ylabel("Vertices in frontier")
    plt.title(f"Frontier split by level ({p.policy})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out1 = os.path.join(out_dir, f"{safe_name(p.policy)}_frontier_by_level.png")
    plt.savefig(out1, dpi=200)
    plt.close()

    frac = []
    for f, g in zip(frontier, gpu_frontier):
        frac.append((g / f) if f != 0 else np.nan)

    plt.figure(figsize=(8, 4))
    plt.plot(lvl, frac, marker="o")
    plt.xlabel("BFS Level")
    plt.ylabel("GPU share of frontier")
    plt.title(f"GPU fraction of frontier by level ({p.policy})")
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out2 = os.path.join(out_dir, f"{safe_name(p.policy)}_gpu_fraction_by_level.png")
    plt.savefig(out2, dpi=200)
    plt.close()

    return (out1, out2)


def plot_median_comparison(adj: ParsedLog, vertex: ParsedLog, out_dir: str) -> str:
    policies = [vertex.policy, adj.policy]
    rust = [vertex.rust_median_ms, adj.rust_median_ms]
    cg_cpu = [vertex.cg_cpu_median_ms, adj.cg_cpu_median_ms]
    cg_gpu = [vertex.cg_gpu_median_ms, adj.cg_gpu_median_ms]

    rust = [np.nan if v is None else v for v in rust]
    cg_cpu = [np.nan if v is None else v for v in cg_cpu]
    cg_gpu = [np.nan if v is None else v for v in cg_gpu]

    x = np.arange(len(policies))
    w = 0.25

    plt.figure(figsize=(9, 4))
    plt.bar(x - w, rust, width=w, label="RustHetGraph (coop median)")
    plt.bar(x, cg_cpu, width=w, label="CGgraph CPU (median)")
    plt.bar(x + w, cg_gpu, width=w, label="CGgraph GPU (median)")

    plt.xticks(x, policies)
    plt.ylabel("Time (ms)")
    plt.title("Median BFS runtime comparison")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(out_dir, "median_comparison.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def require_policy(p: ParsedLog, expected: str):
    if p.policy != expected:
        raise SystemExit(
            f"[ERROR] Expected policy={expected} but parsed policy={p.policy} from file {p.file}. "
            f"Pass the correct log to the correct flag."
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adj", required=True, help="log file for AdjPrefix run")
    ap.add_argument("--vertex", required=True, help="log file for VertexPrefix run")
    ap.add_argument("--out", required=True, help="output directory for plots")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    adj = parse_log(args.adj)
    vertex = parse_log(args.vertex)

    require_policy(adj, "AdjPrefix")
    require_policy(vertex, "VertexPrefix")

    plot_trials(vertex, args.out)
    plot_frontier(vertex, args.out)

    plot_trials(adj, args.out)
    plot_frontier(adj, args.out)

    plot_median_comparison(adj, vertex, args.out)

    def fmt(x):
        return "NA" if x is None else f"{x:.3f}" if isinstance(x, float) else str(x)

    print("\n[SUMMARY] (median over Rust trials; CG medians over 3 runs)")
    for p in [vertex, adj]:
        print(
            f"- {p.policy}: rust_median={fmt(p.rust_median_ms)} ms | "
            f"reached={p.reached_med} | maxd={p.maxd_med} | levels={p.levels_count} | "
            f"m_gpu={p.m_gpu} | edges_on_gpu={p.edges_on_gpu} | "
            f"cg_cpu={fmt(p.cg_cpu_median_ms)} ms | cg_gpu={fmt(p.cg_gpu_median_ms)} ms"
        )

    print(f"\nWrote plots to: {args.out}")


if __name__ == "__main__":
    main()
