use std::collections::VecDeque;
use rust_het_graph::graph::io_bin::load_reordered_csr_from_dir;
use rust_het_graph::graph::stats::validate_csr;
use rust_het_graph::graph::subgraph::SubgraphPolicy;
use rust_het_graph::alg::bfs_coop::CoopBfsPlan;

fn bfs_cpu_dist(g: &rust_het_graph::graph::csr::CsrGraph, root: u32) -> Vec<i32> {
    let n = g.n() as usize;
    let mut dist = vec![-1i32; n];
    let mut q = VecDeque::new();

    let r = root as usize;
    dist[r] = 0;
    q.push_back(root);

    while let Some(u) = q.pop_front() {
        let ui = u as usize;
        let s = g.offsets[ui] as usize;
        let e = g.offsets[ui + 1] as usize;

        let du = dist[ui];
        for &v in &g.dst[s..e] {
            let vi = v as usize;
            if vi < n && dist[vi] < 0 {
                dist[vi] = du + 1;
                q.push_back(v);
            }
        }
    }

    dist
}

fn main() -> cust::error::CudaResult<()> {
    // parse args similarly to your compare bin:
    // <dir> <root> <edge_budget> <per_vertex_cap> --policy ... --debug

    let dir = std::path::PathBuf::from(std::env::args().nth(1).expect("dir required"));
    let root: u32 = std::env::args().nth(2).unwrap_or("0".into()).parse().unwrap();
    let edge_budget: usize = std::env::args().nth(3).unwrap_or("1500000".into()).parse().unwrap();
    let per_vertex_cap: usize = std::env::args().nth(4).unwrap_or("4096".into()).parse().unwrap();

    // minimal policy parse (reuse your existing parse if you want)
    let mut policy = SubgraphPolicy::AdjPrefix;
    let mut debug = false;
    let mut i = 5;
    let argv: Vec<String> = std::env::args().collect();
    while i < argv.len() {
        match argv[i].as_str() {
            "--policy" => {
                policy = SubgraphPolicy::parse(&argv[i + 1]).expect("bad policy");
                i += 2;
            }
            "--debug" => {
                debug = true;
                i += 1;
            }
            _ => panic!("unknown arg: {}", argv[i]),
        }
    }

    let g = load_reordered_csr_from_dir(&dir).expect("load failed");
    validate_csr(&g).expect("CSR invalid");

    println!("[CPU] Running baseline BFS...");
    let dist_cpu = bfs_cpu_dist(&g, root);

    println!("[COOP] Building plan + running coop BFS...");
    let mut plan = CoopBfsPlan::new(&g, edge_budget, per_vertex_cap, policy)?;
    let (_res, dist_coop) = plan.run_with_dist(root, debug)?;

    println!("[CHECK] Comparing distances...");
    let mut mismatches = 0usize;
    let mut first = None;

    for (idx, (&a, &b)) in dist_cpu.iter().zip(dist_coop.iter()).enumerate() {
        if a != b {
            mismatches += 1;
            if first.is_none() {
                first = Some((idx, a, b));
            }
            if mismatches <= 10 {
                println!("  mismatch vid={} cpu={} coop={}", idx, a, b);
            }
        }
    }

    if mismatches == 0 {
        println!("[PASS] dist arrays match exactly.");
    } else {
        let (idx, a, b) = first.unwrap();
        println!("[FAIL] mismatches={} first_vid={} cpu={} coop={}", mismatches, idx, a, b);
    }

    Ok(())
}
