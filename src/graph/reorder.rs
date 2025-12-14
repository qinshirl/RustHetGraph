use std::collections::VecDeque;

use super::csr::CsrGraph;

#[derive(Debug)]
pub struct ReorderResult {
    /// rank[new] = old
    pub rank: Vec<u32>,
    /// old2new[old] = new
    pub old2new: Vec<u32>,
    /// out_degree[old]
    pub out_degree: Vec<u32>,
}

pub fn cggraph_rank_v15(g: &CsrGraph) -> Result<ReorderResult, String> {
    let n = g.n();
    if n == 0 {
        return Ok(ReorderResult {
            rank: vec![],
            old2new: vec![],
            out_degree: vec![],
        });
    }

    // outDegree[old] = offsets[old+1] - offsets[old]
    let mut out_degree = vec![0u32; n];
    for u in 0..n {
        out_degree[u] = g.offsets[u + 1] - g.offsets[u];
    }

    // vertices sorted by (outDegree, id) ascending
    let mut vertices: Vec<u32> = (0u32..(n as u32)).collect();
    vertices.sort_unstable_by(|&a, &b| {
        let da = out_degree[a as usize];
        let db = out_degree[b as usize];
        da.cmp(&db).then_with(|| a.cmp(&b))
    });

    let mut visited = vec![false; n];
    let mut rank: Vec<u32> = Vec::with_capacity(n);

    let mut q: VecDeque<u32> = VecDeque::new();
    let mut nbrs: Vec<u32> = Vec::new(); // reused buffer to reduce allocs

    for &root in &vertices {
        let r = root as usize;
        if visited[r] {
            continue;
        }

        // CGgraph: if degree==0, append directly; else BFS-like expansion
        if out_degree[r] == 0 {
            visited[r] = true;
            rank.push(root);
            continue;
        }

        visited[r] = true;
        q.push_back(root);

        while let Some(v) = q.pop_front() {
            rank.push(v);

            // gather unvisited neighbors of v
            nbrs.clear();
            let start = g.offsets[v as usize] as usize;
            let end = g.offsets[v as usize + 1] as usize;
            for &to in &g.dst[start..end] {
                let ti = to as usize;
                if !visited[ti] {
                    visited[ti] = true;
                    nbrs.push(to);
                }
            }

            // sort neighbors by (outDegree, id) ascending
            nbrs.sort_unstable_by(|&a, &b| {
                let da = out_degree[a as usize];
                let db = out_degree[b as usize];
                da.cmp(&db).then_with(|| a.cmp(&b))
            });

            // push in sorted order
            for &to in &nbrs {
                q.push_back(to);
            }
        }
    }

    // CGgraph reverses rank at the end
    rank.reverse();

    // Build old2new: old2new[rank[new]] = new
    let mut old2new = vec![u32::MAX; n];
    for (new_id, &old_id) in rank.iter().enumerate() {
        old2new[old_id as usize] = new_id as u32;
    }

    // Validate permutation (cheap sanity check; helps catch bugs early)
    validate_permutation(&rank, &old2new)?;

    Ok(ReorderResult {
        rank,
        old2new,
        out_degree,
    })
}

fn validate_permutation(rank: &[u32], old2new: &[u32]) -> Result<(), String> {
    let n = rank.len();
    if old2new.len() != n {
        return Err(format!("old2new len {} != rank len {}", old2new.len(), n));
    }

    let mut seen_old = vec![false; n];
    for &old in rank {
        let oi = old as usize;
        if oi >= n {
            return Err(format!("rank contains out-of-range old id {old}"));
        }
        if seen_old[oi] {
            return Err(format!("rank contains duplicate old id {old}"));
        }
        seen_old[oi] = true;
    }

    let mut seen_new = vec![false; n];
    for (old, &newv) in old2new.iter().enumerate() {
        if newv == u32::MAX {
            return Err(format!("old2new[{old}] was never assigned (u32::MAX)"));
        }
        let ni = newv as usize;
        if ni >= n {
            return Err(format!("old2new[{old}] out of range: {newv}"));
        }
        if seen_new[ni] {
            return Err(format!("old2new maps multiple olds to the same new id {newv}"));
        }
        seen_new[ni] = true;
    }

    Ok(())
}
