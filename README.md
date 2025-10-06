# RustHetGraph: A Rust-Based CPU-GPU Graph Processing Prototype

_Re-implementing the core architectural principles of CGgraph using Rust’s safe and modern systems programming model._

---

## Team Members

**Shirley Qin**
Email: shirley.qin@mail.utoronto.ca

**Guanqun Dong**
Email: guanqun.dong@mail.utoronto.ca

## Motivation

Graph analytics on large-scale datasets, such as social and web networks, increasingly rely on heterogeneous computing systems that combine CPU and GPU resources. While GPUs offer substantial parallelism, their limited global memory and high data-transfer costs over the PCIe bus often restrict performance and scalability.
Recent heterogeneous graph systems, including [**CGgraph**](https://www.vldb.org/pvldb/vol17/p1405-yuan.pdf), demonstrate that a carefully co-designed CPU-GPU framework can outperform both CPU-only and GPU-only approaches by addressing two persistent challenges:

1. **GPU memory over-subscription**, where the full graph cannot fit into device memory, leading to frequent data transfers.
2. **Inefficient CPU-GPU cooperation**, where workloads are statically partitioned, causing underutilization of one processor.

CGgraph introduces several innovations to overcome these limitations:

- **Hardware-oriented graph reordering** to enhance data locality and reduce redundant computation.
- **Size-constrained subgraph extraction**, ensuring that the selected subgraph fits in GPU memory and can be reused across multiple iterations, following the _“load once, use many times”_ principle.
- A **dynamic CPU-GPU co-processing strategy** that balances workloads across processors by invoking the GPU only when beneficial, thereby minimizing unnecessary PCIe transfers.

Despite the demonstrated success of such techniques, **existing implementations are written in C++/CUDA**, with complex memory management and limited accessibility. The Rust ecosystem currently lacks a compact, open-source framework that encapsulates these principles while leveraging Rust’s advantages:

- **Memory and thread safety** without garbage collection, reducing concurrency-related errors.
- **Zero-cost abstractions** for direct control of memory layout and computation scheduling.
- **Modern parallel and GPU libraries**, such as `rayon` for CPU parallelism and `wgpu` for portable GPU compute.

This project aims to fill this gap by developing RustHetGraph, a Rust-based re-implementation of CGgraph’s core ideas graph reordering, size-constrained subgraph extraction, and CPU-GPU co-processing policy in a simplified framework.
The resulting system will serve both as a platform for heterogeneous graph computation and as an exploratory benchmark to assess how Rust’s safety and concurrency model compare to traditional C++/CUDA implementations in this domain.

## Objective and Key Features

### Objective

The objective of this project is to design and implement RustHetGraph, a Rust-based prototype that re-implements the key architectural concepts of CGgraph for heterogeneous CPU-GPU graph processing.
Specifically, the project seeks to reproduce CGgraph’s hardware-oriented graph reordering, size-constrained subgraph extraction, and adaptive CPU-GPU co-processing policy in a simplified yet analytically meaningful form.

RustHetGraph aims to:

1. Explore how Rust’s safe concurrency model and fine-grained memory control can support high-performance graph analytics, and
2. Evaluate whether similar performance behaviors and design principles observed in CGgraph can be achieved or approximated within Rust’s programming model.

The ultimate goal is to create an prototype that enables both conceptual and empirical comparison between CGgraph (C++/CUDA) and RustHetGraph (Rust + Rayon / optional wgpu).

---

### Key Features

#### 1. Hardware-Oriented Graph Reordering

Implements a vertex reordering strategy inspired by CGgraph’s hardware-aware design.
Vertices with high connectivity are prioritized, while sink vertices are deferred to the end of the adjacency list.
This ordering improves cache locality and prepares the graph for efficient subgraph extraction.

#### 2. Size-Constrained Subgraph Extraction

Implements the “load once, use multiple times” principle by extracting a GPU-resident subgraph(G') that captures the most active portion of the graph.
The subgraph size is determined by a configurable memory threshold representing the GPU’s capacity.
When GPU integration is unavailable, G' can be simulated on CPU threads to verify scheduling behavior.

#### 3. Adaptive CPU-GPU Co-Processing Policy

Develops a simplified version of CGgraph’s per-iteration GPU invocation policy.
Using tunable thresholds (active edge count and coverage ratio), the system dynamically decides whether to engage GPU computation or keep execution on the CPU, minimizing unnecessary data transfers and idle cycles.

#### 4. Algorithm Evaluation and Comparative Analysis

Integrates representative graph algorithms (BFS, SSSP,PR,WCC) to evaluate performance under varying configurations encompassing the original and reordered graph layouts, the presence or absence of GPU subgraph extraction, and the use of static versus adaptive scheduling policies.

Execution results will be compared with those obtained from running CGgraph on equivalent datasets, enabling an assessment of how Rust’s safety-oriented design affects performance, scalability, and scheduling efficiency relative to the original C++/CUDA implementation.

## Tentative Plan

The project will be carried out over 10 weeks, and the schedule below outlines detailed weekly goals and deliverables.

| **Week**                                              | **Shirley Qin**                                                                                                                                                            | **Guanqun Dong**                                                                                                                                                                                                                 | **Deliverables**                                                                                       |
| ----------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| **1 - Environment Setup and Dataset Preparation**     | - Configure Rust toolchain, dependencies (`rayon`, `criterion`, `serde`, `clap`).<br />- Establish project structure (graph IO, reorder, etc.)                             | - Collect open datasets <br />- Write data-conversion scripts for standardized CSV edge lists.                                                                                                                                   | Project repository initialized with working I/O and test datasets.                                     |
| **2 - Baseline Graph Representation**                 | - Implement Compressed Sparse Row (CSR) representation with safe memory ownership.<br />- Test adjacency access and traversal correctness.                                 | - Develop CLI for loading graphs and reporting graph statistics (vertex count, edge count, degree distribution).                                                                                                                 | Verified CSR-based graph engine and command-line graph loader.                                         |
| **3 - Baseline Algorithms and Profiling**             | - Implement sequential BFS and SSSP as computational baselines.                                                                                                            | - Integrate `criterion` benchmarking to capture runtime and memory metrics.<br />- Produce initial performance summary.                                                                                                          | Verified BFS/SSSP algorithms with reproducible timing profiles.                                        |
| **4 - Hardware-Oriented Graph Reordering**            | - Implement degree-based vertex reordering inspired by CGgraph.<br />- Validate adjacency integrity after reindexing.                                                      | - Generate statistics and visualizations (degree distribution, neighbor locality).                                                                                                                                               | Working reordering module with verified correctness and improved locality analysis.                    |
| **5 - Size-Constrained Subgraph Extraction**          | - Implement subgraph extraction selecting active vertices and edges under a configurable GPU-memory threshold (`--budget-mb`).                                             | - Test extraction on multiple datasets; compute coverage ratio and memory estimate.<br />- Output diagnostic JSON summary.                                                                                                       | Functional subgraph (G′) extraction producing coverage and size metrics.                               |
| **6 - Reuse Mechanism and Iteration Control**         | - Implement “load once, reuse many times” mechanism allowing multiple iterations over (G′) without reload.<br />- Ensure ownership and borrowing safety across iterations. | - Verify correctness across repeated runs; add CLI options for iteration count and log output.                                                                                                                                   | Verified subgraph reuse pipeline ensuring consistent multi-iteration behavior.                         |
| **7 - Adaptive CPU-GPU Scheduling Policy**            | - Implement adaptive scheduling policy using thresholds (τ) (active edges) and (θ) (coverage in (G′)) for CPU vs. hybrid decision-making.                                  | - Implement CPU-based GPU emulator with `rayon` thread pools; simulate GPU compute and transfer latency.                                                                                                                         | Operational adaptive scheduler demonstrating workload partitioning and threshold switching.            |
| **8 - Parallelization and Optional GPU Integration**  | - Parallelize BFS/SSSP with `rayon` parallel iterators.<br />- Begin optional integration of **`wgpu`** compute-shader kernel for small-scale GPU testing.                 | - Configure builds for optional GPU mode; run timing comparisons (single-thread, multithread, GPU).                                                                                                                              | Optimized parallel execution and preliminary GPU-path prototype.                                       |
| **9 - Algorithm Evaluation and Comparative Analysis** | - Implement PR and WCC to broaden evaluation coverage.- Prepare result visualization scripts.                                                                              | - Execute experiments under varying configurations (original vs. reordered, with/without (G′), static vs. adaptive policy).<br />- Compile tables and plots.<br />- Compare results with CGgraph’s published data qualitatively. | Consolidated benchmark dataset and draft figures for report.                                           |
| **10 - Documentation, Presentation, and Submission**  | - Finalize documentation of architecture, design rationale, and key implementation details.<br />- Code review and prepare final README.                                   | - Prepare slide presentation and demo video.<br />- Verify reproducibility of results and finalize submission.                                                                                                                   | Complete, documented submission package including source code, benchmarks, and presentation materials. |

---

### Feasibility and Risk Management

All core components — graph reordering, subgraph extraction, and adaptive scheduling — are planned for completion by Week 8, leaving the final two weeks for evaluation and documentation.
If GPU integration via `wgpu` is not fully achieved, the CPU-based emulator will ensure that hybrid scheduling and reuse behavior can still be demonstrated.
