use cust::error::CudaResult;
use cust::prelude::*;
use std::time::Instant;

use crate::gpu::runtime::FRONTIER_EXPAND_PTX;
use crate::gpu::subgraph_device::GpuSubgraphDevice;
use crate::graph::csr::CsrGraph;
use crate::graph::subgraph::{build_gpu_subgraph, GpuSubgraph, SubgraphPolicy};

#[derive(Debug, Clone)]
pub struct BfsCoopResult {
    pub reached: u64,
    pub maxd: i32,
    pub levels: u32,
    pub total_ms: f64,

    pub policy: SubgraphPolicy,
    pub m_gpu: usize,
    pub edges_on_gpu: f64,
}

fn bfs_stats(dist: &[i32]) -> (u64, i32) {
    let mut reached = 0u64;
    let mut maxd = -1;
    for &d in dist {
        if d >= 0 {
            reached += 1;
            if d > maxd {
                maxd = d;
            }
        }
    }
    (reached, maxd)
}

pub struct CoopBfsPlan<'a> {
    g: &'a CsrGraph,
    policy: SubgraphPolicy,

    sub_host: GpuSubgraph,
    sub_dev: GpuSubgraphDevice,

    _ctx: Context,
    stream: Stream,
    module: Module,

    dist_d: DeviceBuffer<i32>,
    out_d: DeviceBuffer<u32>,
    out_len_d: DeviceBuffer<u32>,

    dist_h: Vec<i32>,
    frontier_h: Vec<u32>,
    out_h: Vec<u32>,

    gpu_n: u32,
    gpu_dst_len: u32,
    out_cap: u32,

    n: usize,
    m_gpu: usize,
    edges_on_gpu: f64,
}

impl<'a> CoopBfsPlan<'a> {
    pub fn new(
        g: &'a CsrGraph,
        edge_budget: usize,
        per_vertex_cap: usize,
        policy: SubgraphPolicy,
    ) -> CudaResult<Self> {
        let n = g.n();

        let sub_host = build_gpu_subgraph(g, policy, edge_budget, per_vertex_cap);
        let m_gpu = sub_host.m_gpu();
        let edges_on_gpu = if g.m() == 0 {
            0.0
        } else {
            (m_gpu as f64) / (g.m() as f64)
        };

        let gpu_n = sub_host.gpu_offsets().len().saturating_sub(1) as u32;
        let gpu_dst_len = sub_host.gpu_dst().len() as u32;

        let _ctx = cust::quick_init()?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
        let module = Module::from_ptx(FRONTIER_EXPAND_PTX, &[])?;

        let sub_dev = GpuSubgraphDevice::from_host(&sub_host)?;

        let dist_h = vec![-1; n];
        let dist_d = DeviceBuffer::from_slice(&dist_h)?;

        const OUT_CAP_ELEMS: usize = 50_000_000;
        let out_d = unsafe { DeviceBuffer::<u32>::uninitialized(OUT_CAP_ELEMS)? };
        let out_len_d = DeviceBuffer::from_slice(&[0u32])?;
        let out_cap = OUT_CAP_ELEMS as u32;

        Ok(Self {
            g,
            policy,
            sub_host,
            sub_dev,
            _ctx,
            stream,
            module,
            dist_d,
            out_d,
            out_len_d,
            dist_h,
            frontier_h: Vec::with_capacity(1024),
            out_h: Vec::new(),
            gpu_n,
            gpu_dst_len,
            out_cap,
            n,
            m_gpu,
            edges_on_gpu,
        })
    }

    fn reset(&mut self, root: u32) -> CudaResult<()> {
        self.dist_h.fill(-1);
        if (root as usize) < self.n {
            self.dist_h[root as usize] = 0;
        }
        self.dist_d.copy_from(&self.dist_h)?;

        self.frontier_h.clear();
        self.out_h.clear();
        Ok(())
    }

    pub fn run_with_dist(
        &mut self,
        root: u32,
        debug: bool,
    ) -> CudaResult<(BfsCoopResult, Vec<i32>)> {
        let res = self.run(root, debug)?;
        Ok((res, self.dist_h.clone()))
    }

    pub fn dist_host(&self) -> &[i32] {
        &self.dist_h
    }

    pub fn run(&mut self, root: u32, debug: bool) -> CudaResult<BfsCoopResult> {
        self.reset(root)?;

        let t0 = Instant::now();
        self.frontier_h.push(root);

        let edge_cut_opt = self.sub_host.gpu_edge_cut_opt();
        let gpu_offsets = self.sub_host.gpu_offsets();

        let mut frontier_d: Option<DeviceBuffer<u32>> = None;

        let mut gpu_dead_levels: usize = 0;
        const GPU_DEAD_THRESHOLD: usize = 3;
        let mut gpu_enabled: bool = true;

        if debug {
            println!(
                "[DEBUG] gpu_n(offsets domain) = {}, gpu_dst_len = {}, out_cap = {}",
                self.gpu_n, self.gpu_dst_len, self.out_cap
            );
        }

        let mut level: i32 = 0;

        while !self.frontier_h.is_empty() {
            let mut gpu_frontier: Vec<u32> = Vec::new();
            let mut cpu_frontier: Vec<u32> = Vec::new();

            if !gpu_enabled {
                cpu_frontier.extend_from_slice(&self.frontier_h);
            } else {
                match self.policy {
                    SubgraphPolicy::AdjPrefix => {
                        for &u in &self.frontier_h {
                            if u < self.gpu_n {
                                let ui = u as usize;
                                let start = gpu_offsets[ui];
                                let end = gpu_offsets[ui + 1];
                                if end > start {
                                    gpu_frontier.push(u);
                                } else {
                                    cpu_frontier.push(u);
                                }
                            } else {
                                cpu_frontier.push(u);
                            }
                        }
                    }
                    SubgraphPolicy::VertexPrefix => {
                        for &u in &self.frontier_h {
                            if u < self.gpu_n {
                                gpu_frontier.push(u);
                            } else {
                                cpu_frontier.push(u);
                            }
                        }
                    }
                }
            }

            let mut gpu_out: Vec<u32> = Vec::new();

            if gpu_enabled && !gpu_frontier.is_empty() {
                let k_usize = gpu_frontier.len();
                let k = k_usize as u32;

                if frontier_d.as_ref().map_or(true, |b| b.len() != k_usize) {
                    frontier_d = Some(unsafe { DeviceBuffer::<u32>::uninitialized(k_usize)? });
                }
                let frontier_buf = frontier_d.as_mut().unwrap();

                frontier_buf.copy_from(&gpu_frontier)?;

                self.out_len_d.copy_from(&[0u32])?;

                let block = 256u32;
                let grid = (k + block - 1) / block;

                let module = &self.module;
                let stream = &self.stream;

                unsafe {
                    launch!(
                        module.frontier_expand_subgraph_only<<<grid, block, 0, stream>>>(
                            frontier_buf.as_device_ptr(),
                            k,
                            self.sub_dev.offsets_d.as_device_ptr(),
                            self.sub_dev.dst_d.as_device_ptr(),
                            self.gpu_n,
                            self.gpu_dst_len,
                            self.out_d.as_device_ptr(),
                            self.out_len_d.as_device_ptr(),
                            self.out_cap
                        )
                    )?;
                }

                self.stream.synchronize()?;

                let mut out_len_host = [0u32];
                self.out_len_d.copy_to(&mut out_len_host)?;
                let out_len = out_len_host[0] as usize;

                if out_len > 0 {
                    gpu_dead_levels = 0;

                    let out_len = out_len.min(self.out_d.len());
                    self.out_h.resize(out_len, 0);

                    self.out_d.index(..out_len).copy_to(&mut self.out_h)?;

                    gpu_out.extend_from_slice(&self.out_h);
                } else {
                    gpu_dead_levels += 1;

                    if gpu_dead_levels >= GPU_DEAD_THRESHOLD {
                        gpu_enabled = false;

                        cpu_frontier.extend_from_slice(&gpu_frontier);
                        gpu_frontier.clear();
                    }
                }
            }

            let mut cpu_cand: Vec<u32> = Vec::new();

            for &u in &cpu_frontier {
                let ui = u as usize;
                let s = self.g.offsets[ui] as usize;
                let e = self.g.offsets[ui + 1] as usize;
                cpu_cand.extend_from_slice(&self.g.dst[s..e]);
            }

            if gpu_enabled {
                if let (SubgraphPolicy::AdjPrefix, Some(edge_cut)) = (self.policy, edge_cut_opt.as_ref())
                {
                    for &u in &gpu_frontier {
                        let ui = u as usize;
                        let c = edge_cut[ui] as usize;
                        let e = self.g.offsets[ui + 1] as usize;
                        if c < e {
                            cpu_cand.extend_from_slice(&self.g.dst[c..e]);
                        }
                    }
                }
            }

            let mut next: Vec<u32> = Vec::new();

            for v in cpu_cand.into_iter().chain(gpu_out.into_iter()) {
                let vi = v as usize;
                if vi < self.n && self.dist_h[vi] < 0 {
                    self.dist_h[vi] = level + 1;
                    next.push(v);
                }
            }

            if debug {
                println!(
                    "[DEBUG] lvl={} frontier={} gpu_frontier={} cpu_frontier={} next={} gpu_enabled={}",
                    level,
                    self.frontier_h.len(),
                    gpu_frontier.len(),
                    cpu_frontier.len(),
                    next.len(),
                    gpu_enabled
                );
            }

            self.frontier_h = next;
            level += 1;

            if level > 100_000 {
                break;
            }
        }

        let (reached, maxd) = bfs_stats(&self.dist_h);
        let total_ms = t0.elapsed().as_secs_f64() * 1000.0;

        Ok(BfsCoopResult {
            reached,
            maxd,
            levels: (maxd.max(0) as u32) + 1,
            total_ms,
            policy: self.policy,
            m_gpu: self.m_gpu,
            edges_on_gpu: self.edges_on_gpu,
        })
    }
}

pub fn bfs_coop_gpu_cpu(
    g: &CsrGraph,
    root: u32,
    edge_budget: usize,
    per_vertex_cap: usize,
    policy: SubgraphPolicy,
    debug: bool,
) -> CudaResult<BfsCoopResult> {
    let mut plan = CoopBfsPlan::new(g, edge_budget, per_vertex_cap, policy)?;
    plan.run(root, debug)
}
