use super::csr::CsrGraph;

pub fn permute_csr_cggraph_v15(
    g: &CsrGraph,
    rank: &[u32],      // rank[new] = old
    old2new: &[u32],   // old2new[old] = new
) -> Result<CsrGraph, String> {
    let n = g.n();
    if rank.len() != n || old2new.len() != n {
        return Err(format!(
            "permute: rank len {} / old2new len {} must equal n {}",
            rank.len(),
            old2new.len(),
            n
        ));
    }

    // Compute new offsets by preserving each vertex's out-degree:
    // outDegree_new[new_u] = outDegree_old[old_u]
    let mut offsets_new = vec![0u32; n + 1];

    let mut acc: u64 = 0;
    for new_u in 0..n {
        let old_u = rank[new_u] as usize;
        let deg = (g.offsets[old_u + 1] - g.offsets[old_u]) as u64;
        acc += deg;
        if acc > u32::MAX as u64 {
            return Err("permute: edge count overflowed u32 offsets".into());
        }
        offsets_new[new_u + 1] = acc as u32;
    }

    let m_new = offsets_new[n] as usize;
    if m_new != g.dst.len() {
        // In CGgraph reorder, edge count is unchanged; if this fails, something is wrong.
        return Err(format!(
            "permute: m_new {} != original m {}",
            m_new,
            g.dst.len()
        ));
    }

    let mut dst_new = vec![0u32; m_new];
    let mut w_new: Option<Vec<u32>> = g.w.as_ref().map(|_| vec![0u32; m_new]);

    // Fill dst_new (and w_new if present) in the new CSR layout
    for new_u in 0..n {
        let old_u = rank[new_u] as usize;

        let old_start = g.offsets[old_u] as usize;
        let old_end = g.offsets[old_u + 1] as usize;

        let new_start = offsets_new[new_u] as usize;

        // Copy & remap adjacency list
        for (k, old_edge_idx) in (old_start..old_end).enumerate() {
            let old_v = g.dst[old_edge_idx] as usize;
            let new_v = old2new[old_v]; // new id
            dst_new[new_start + k] = new_v;

            if let (Some(w_old), Some(w_out)) = (g.w.as_ref(), w_new.as_mut()) {
                w_out[new_start + k] = w_old[old_edge_idx];
            }
        }
    }

    Ok(CsrGraph {
        offsets: offsets_new,
        dst: dst_new,
        w: w_new,
    })
}
