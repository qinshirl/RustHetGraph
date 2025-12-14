use super::csr::CsrGraph;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubgraphPolicy {
    AdjPrefix,
    VertexPrefix,
}

impl SubgraphPolicy {
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "adj" | "adjprefix" | "adj-prefix" | "prefix" => Some(SubgraphPolicy::AdjPrefix),
            "vertex" | "vertexprefix" | "vertex-prefix" => Some(SubgraphPolicy::VertexPrefix),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SubgraphMetrics {
    pub policy: SubgraphPolicy,
    pub n: usize,
    pub m: usize,
    pub m_gpu: usize,
    pub edges_on_gpu: f64,
    pub cut_vertex: Option<u32>,
}

impl SubgraphMetrics {
    pub fn new(policy: SubgraphPolicy, n: usize, m: usize, m_gpu: usize, cut_vertex: Option<u32>) -> Self {
        let edges_on_gpu = if m == 0 { 0.0 } else { (m_gpu as f64) / (m as f64) };
        Self { policy, n, m, m_gpu, edges_on_gpu, cut_vertex }
    }
}

#[derive(Debug)]
pub struct GpuSubgraphPrefix {
    pub gpu_offsets: Vec<u32>,
    pub gpu_dst: Vec<u32>,
    pub gpu_edge_cut: Vec<u32>,
}

// pub fn build_gpu_subgraph_prefix(
//     g: &CsrGraph,
//     edge_budget: usize,
//     per_vertex_cap: usize,
// ) -> GpuSubgraphPrefix {
//     let n = g.n();
//     let m = g.m();

//     if n == 0 {
//         return GpuSubgraphPrefix {
//             gpu_offsets: vec![0],
//             gpu_dst: vec![],
//             gpu_edge_cut: vec![],
//         };
//     }

//     let edge_budget = edge_budget.min(m);

//     let mut gpu_offsets: Vec<u32> = vec![0u32; n + 1];
//     let mut gpu_edge_cut: Vec<u32> = vec![0u32; n];
//     let mut gpu_dst: Vec<u32> = Vec::with_capacity(edge_budget);

//     let mut used: usize = 0;

//     for u in 0..n {
//         let start = g.offsets[u] as usize;
//         let end = g.offsets[u + 1] as usize;
//         let deg = end - start;

//         let mut take: usize = 0;

//         if used < edge_budget && deg > 0 && per_vertex_cap > 0 {
//             let remaining = edge_budget - used;
//             take = deg.min(per_vertex_cap).min(remaining);

//             gpu_dst.extend_from_slice(&g.dst[start..start + take]);
//             used += take;
//         }

//         gpu_edge_cut[u] = (start + take) as u32;
//         gpu_offsets[u + 1] = used as u32;
//     }

//     debug_assert_eq!(gpu_offsets[n] as usize, gpu_dst.len());

//     GpuSubgraphPrefix {
//         gpu_offsets,
//         gpu_dst,
//         gpu_edge_cut,
//     }
// }

pub fn build_gpu_subgraph_prefix(
    g: &CsrGraph,
    edge_budget: usize,
    per_vertex_cap: usize,
) -> GpuSubgraphPrefix {
    let n = g.n();
    let m = g.m();

    if n == 0 {
        return GpuSubgraphPrefix {
            gpu_offsets: vec![0],
            gpu_dst: vec![],
            gpu_edge_cut: vec![],
        };
    }

    let edge_budget = edge_budget.min(m);
    if edge_budget == 0 || per_vertex_cap == 0 {
        // No GPU edges at all.
        return GpuSubgraphPrefix {
            gpu_offsets: vec![0u32; n + 1],
            gpu_dst: vec![],
            gpu_edge_cut: g.offsets[..n].to_vec(), // cut == start
        };
    }

    // We build per-vertex "take" counts first, then materialize gpu_dst in one pass.
    let mut take_per_v: Vec<usize> = vec![0; n];

    // Round-robin chunk size. Small chunks spread coverage widely.
    // You can tune this; 32/64/128 are reasonable starting points.
    const CHUNK: usize = 64;

    let mut used: usize = 0;
    let mut progressed = true;

    // Multi-pass allocation: give each vertex a small chunk per pass until budget is exhausted.
    while used < edge_budget && progressed {
        progressed = false;

        for u in 0..n {
            if used >= edge_budget {
                break;
            }

            let start = g.offsets[u] as usize;
            let end = g.offsets[u + 1] as usize;
            let deg = end - start;
            if deg == 0 {
                continue;
            }

            // How many edges this vertex is allowed to place on GPU total.
            let cap = per_vertex_cap.min(deg);

            // How many it already has.
            let cur = take_per_v[u];
            if cur >= cap {
                continue;
            }

            let remaining_vertex = cap - cur;
            let remaining_budget = edge_budget - used;

            let add = remaining_vertex.min(CHUNK).min(remaining_budget);
            if add == 0 {
                continue;
            }

            take_per_v[u] = cur + add;
            used += add;
            progressed = true;
        }
    }

    // Now materialize gpu_offsets/gpu_dst/gpu_edge_cut using take_per_v.
    let mut gpu_offsets: Vec<u32> = vec![0u32; n + 1];
    let mut gpu_edge_cut: Vec<u32> = vec![0u32; n];
    let mut gpu_dst: Vec<u32> = Vec::with_capacity(used);

    let mut prefix: usize = 0;
    for u in 0..n {
        let start = g.offsets[u] as usize;
        let t = take_per_v[u];

        if t > 0 {
            gpu_dst.extend_from_slice(&g.dst[start..start + t]);
            prefix += t;
        }

        gpu_edge_cut[u] = (start + t) as u32;
        gpu_offsets[u + 1] = prefix as u32;
    }

    debug_assert_eq!(gpu_offsets[n] as usize, gpu_dst.len());

    GpuSubgraphPrefix {
        gpu_offsets,
        gpu_dst,
        gpu_edge_cut,
    }
}


#[derive(Debug)]
pub struct GpuSubgraphVertexPrefix {
    pub cut_vertex: u32,
    pub gpu_offsets: Vec<u32>,
    pub gpu_dst: Vec<u32>,
}

pub fn build_gpu_subgraph_vertex_prefix(g: &CsrGraph, edge_budget: usize) -> GpuSubgraphVertexPrefix {
    let n = g.n();
    if n == 0 {
        return GpuSubgraphVertexPrefix {
            cut_vertex: 0,
            gpu_offsets: vec![0],
            gpu_dst: vec![],
        };
    }

    let budget_u32 = edge_budget.min(g.m()) as u32;

    let mut lo: usize = 0;
    let mut hi: usize = n;
    while lo < hi {
        let mid = (lo + hi + 1) / 2;
        if g.offsets[mid] <= budget_u32 {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    let k = lo;
    let m_gpu = g.offsets[k] as usize;

    let gpu_dst = g.dst[..m_gpu].to_vec();

    let mut gpu_offsets = g.offsets.clone();
    let clamp = g.offsets[k];
    for v in (k + 1)..=n {
        gpu_offsets[v] = clamp;
    }

    GpuSubgraphVertexPrefix {
        cut_vertex: k as u32,
        gpu_offsets,
        gpu_dst,
    }
}

#[derive(Debug)]
pub enum GpuSubgraph {
    AdjPrefix(GpuSubgraphPrefix),
    VertexPrefix(GpuSubgraphVertexPrefix),
}

impl GpuSubgraph {
    pub fn gpu_offsets(&self) -> &[u32] {
        match self {
            GpuSubgraph::AdjPrefix(s) => &s.gpu_offsets,
            GpuSubgraph::VertexPrefix(s) => &s.gpu_offsets,
        }
    }

    pub fn gpu_dst(&self) -> &[u32] {
        match self {
            GpuSubgraph::AdjPrefix(s) => &s.gpu_dst,
            GpuSubgraph::VertexPrefix(s) => &s.gpu_dst,
        }
    }

    pub fn gpu_edge_cut_opt(&self) -> Option<&[u32]> {
        match self {
            GpuSubgraph::AdjPrefix(s) => Some(&s.gpu_edge_cut),
            GpuSubgraph::VertexPrefix(_) => None,
        }
    }

    pub fn cut_vertex_opt(&self) -> Option<u32> {
        match self {
            GpuSubgraph::AdjPrefix(_) => None,
            GpuSubgraph::VertexPrefix(s) => Some(s.cut_vertex),
        }
    }

    pub fn m_gpu(&self) -> usize {
        self.gpu_dst().len()
    }

    pub fn gpu_deg(&self, u: usize) -> u32 {
        let offs = self.gpu_offsets();
        offs[u + 1] - offs[u]
    }

    pub fn vertex_is_on_gpu(&self, u: u32) -> bool {
        match self {
            GpuSubgraph::AdjPrefix(s) => {
                let ui = u as usize;
                (s.gpu_offsets[ui + 1] - s.gpu_offsets[ui]) > 0
            }
            GpuSubgraph::VertexPrefix(s) => u < s.cut_vertex,
        }
    }

    pub fn metrics(&self, g: &CsrGraph, policy: SubgraphPolicy) -> SubgraphMetrics {
        SubgraphMetrics::new(policy, g.n(), g.m(), self.m_gpu(), self.cut_vertex_opt())
    }
}

pub fn build_gpu_subgraph(
    g: &CsrGraph,
    policy: SubgraphPolicy,
    edge_budget: usize,
    per_vertex_cap: usize,
) -> GpuSubgraph {
    match policy {
        SubgraphPolicy::AdjPrefix => {
            let s = build_gpu_subgraph_prefix(g, edge_budget, per_vertex_cap);
            GpuSubgraph::AdjPrefix(s)
        }
        SubgraphPolicy::VertexPrefix => {
            let s = build_gpu_subgraph_vertex_prefix(g, edge_budget);
            GpuSubgraph::VertexPrefix(s)
        }
    }
}
