use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::graph::csr::CsrGraph;

pub fn dijkstra_sssp(g: &CsrGraph, root_new: u32) -> Result<Vec<u64>, String> {
    let w = g.w.as_ref().ok_or("SSSP requires weights, but g.w is None")?;

    let n = g.n();
    let mut dist = vec![u64::MAX; n];

    let r = root_new as usize;
    dist[r] = 0;

    let mut pq: BinaryHeap<(Reverse<u64>, u32)> = BinaryHeap::new();
    pq.push((Reverse(0), root_new));

    while let Some((Reverse(dv), v)) = pq.pop() {
        if dv != dist[v as usize] {
            continue; // stale entry
        }

        let start = g.offsets[v as usize] as usize;
        let end = g.offsets[v as usize + 1] as usize;

        for ei in start..end {
            let to = g.dst[ei];
            let wt = w[ei] as u64;

            let nd = dv.saturating_add(wt);
            let ti = to as usize;

            if nd < dist[ti] {
                dist[ti] = nd;
                pq.push((Reverse(nd), to));
            }
        }
    }

    Ok(dist)
}
