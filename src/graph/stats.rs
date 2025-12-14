use super::csr::CsrGraph;

pub fn validate_csr(g: &CsrGraph) -> Result<(), String> {
    let n = g.n();
    if g.offsets.len() != n + 1 {
        return Err("offsets length must be n+1".into());
    }
    if g.offsets.is_empty() || g.offsets[0] != 0 {
        return Err("offsets[0] must be 0".into());
    }
    let m = g.m();
    let last = g.offsets[n] as usize;
    if last != m {
        return Err(format!("offsets[n] = {last}, but dst.len() = {m}"));
    }
    for i in 0..n {
        if g.offsets[i] > g.offsets[i + 1] {
            return Err(format!("offsets not non-decreasing at i={i}"));
        }
    }
    if let Some(w) = &g.w {
        if w.len() != m {
            return Err(format!("weights len {} != edges {}", w.len(), m));
        }
    }
    Ok(())
}

pub fn degree_summary(g: &CsrGraph) -> (u32, u32, f64) {
    let n = g.n();
    if n == 0 {
        return (0, 0, 0.0);
    }
    let mut min_d = u32::MAX;
    let mut max_d = 0u32;
    let mut sum: u64 = 0;

    for u in 0..n {
        let d = g.offsets[u + 1] - g.offsets[u];
        min_d = min_d.min(d);
        max_d = max_d.max(d);
        sum += d as u64;
    }

    (min_d, max_d, sum as f64 / n as f64)
}
