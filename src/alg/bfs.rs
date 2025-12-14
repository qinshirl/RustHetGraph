use std::collections::VecDeque;
use crate::graph::csr::CsrGraph;

/// BFS on directed CSR graph.
pub fn bfs(g: &CsrGraph, root_new: u32) -> Vec<i32> {
    let n = g.n();
    let mut dist = vec![-1i32; n];

    let r = root_new as usize;
    dist[r] = 0;

    let mut q = VecDeque::new();
    q.push_back(root_new);

    while let Some(v) = q.pop_front() {
        let dv = dist[v as usize];
        let start = g.offsets[v as usize] as usize;
        let end = g.offsets[v as usize + 1] as usize;

        for &to in &g.dst[start..end] {
            let ti = to as usize;
            if dist[ti] == -1 {
                dist[ti] = dv + 1;
                q.push_back(to);
            }
        }
    }

    dist
}
