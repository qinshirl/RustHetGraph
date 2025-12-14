use std::fs::File;
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

#[derive(Debug)]
struct Config {
    edge_path: PathBuf,
    out_dir: PathBuf,
    graph_name: String,
    undirected: bool,
    weight: u32,
}

fn usage_and_exit() -> ! {
    eprintln!(
        r#"Usage:
  cargo run --release --bin edge_list_to_csr -- <edge_txt> <out_dir> <graph_name> [--undirected] [--weight <u32>]
  - output style:
      <graph>_csrOffset_u32.bin
      <graph>_csrDest_u32.bin
      <graph>_csrWeight_u32.bin
"#
    );
    std::process::exit(2);
}

fn parse_args() -> Config {
    let mut args = std::env::args().skip(1);

    let edge_path = args.next().map(PathBuf::from).unwrap_or_else(|| usage_and_exit());
    let out_dir = args.next().map(PathBuf::from).unwrap_or_else(|| usage_and_exit());
    let graph_name = args.next().unwrap_or_else(|| usage_and_exit());

    let mut undirected = false;
    let mut weight: u32 = 1;

    while let Some(tok) = args.next() {
        match tok.as_str() {
            "--undirected" => undirected = true,
            "--weight" => {
                let v = args.next().unwrap_or_else(|| usage_and_exit());
                weight = v.parse().unwrap_or_else(|_| usage_and_exit());
            }
            _ => usage_and_exit(),
        }
    }

    Config { edge_path, out_dir, graph_name, undirected, weight }
}

fn parse_edge(line: &str) -> Option<(u32, u32)> {
    let s = line.trim();
    if s.is_empty() || s.starts_with('#') {
        return None;
    }
    let mut it = s.split_whitespace();
    let a = it.next()?;
    let b = it.next()?;
    let u: u32 = a.parse().ok()?;
    let v: u32 = b.parse().ok()?;
    Some((u, v))
}

fn write_u32_bin(path: &Path, xs: &[u32]) -> io::Result<()> {
    let mut w = BufWriter::new(File::create(path)?);
    for &x in xs {
        w.write_all(&x.to_le_bytes())?;
    }
    w.flush()?;
    Ok(())
}

fn main() -> io::Result<()> {
    let cfg = parse_args();

    std::fs::create_dir_all(&cfg.out_dir)?;

    eprintln!("[CFG] edge_path={:?}", cfg.edge_path);
    eprintln!("[CFG] out_dir={:?}", cfg.out_dir);
    eprintln!("[CFG] graph_name={}", cfg.graph_name);
    eprintln!("[CFG] undirected={}", cfg.undirected);
    eprintln!("[CFG] weight={}", cfg.weight);

    eprintln!("[PASS0] scanning for max node id...");
    let f = File::open(&cfg.edge_path)?;
    let r = BufReader::new(f);

    let mut max_id: u32 = 0;
    let mut lines: u64 = 0;
    for line in r.lines() {
        let line = line?;
        if let Some((u, v)) = parse_edge(&line) {
            max_id = max_id.max(u).max(v);
            lines += 1;
        }
    }
    let v = (max_id as usize) + 1;
    eprintln!("[PASS0] lines={} max_id={} => V={}", lines, max_id, v);

    eprintln!("[PASS1] counting degrees...");
    let f = File::open(&cfg.edge_path)?;
    let r = BufReader::new(f);

    let mut deg: Vec<u32> = vec![0u32; v];
    let mut e_cnt: u64 = 0;

    for line in r.lines() {
        let line = line?;
        if let Some((u, vv)) = parse_edge(&line) {
            deg[u as usize] = deg[u as usize].wrapping_add(1);
            e_cnt += 1;

            if cfg.undirected {
                deg[vv as usize] = deg[vv as usize].wrapping_add(1);
                e_cnt += 1;
            }
        }
    }

    let e = e_cnt as usize;
    eprintln!("[PASS1] E={} (undirected={})", e, cfg.undirected);

    eprintln!("[BUILD] building offsets...");
    let mut offsets: Vec<u32> = vec![0u32; v + 1];
    let mut sum: u64 = 0;
    for i in 0..v {
        offsets[i] = sum as u32;
        sum += deg[i] as u64;
    }
    offsets[v] = sum as u32;

    let mut cursor: Vec<u32> = offsets[..v].to_vec();

    eprintln!("[ALLOC] allocating dest/weight arrays...");
    let mut dest: Vec<u32> = vec![0u32; e];
    let mut wts: Vec<u32> = vec![cfg.weight; e];

    eprintln!("[PASS2] filling dest...");
    let f = File::open(&cfg.edge_path)?;
    let r = BufReader::new(f);

    for line in r.lines() {
        let line = line?;
        if let Some((u, vv)) = parse_edge(&line) {
            let p = cursor[u as usize] as usize;
            dest[p] = vv;
            wts[p] = cfg.weight;
            cursor[u as usize] += 1;

            if cfg.undirected {
                let p2 = cursor[vv as usize] as usize;
                dest[p2] = u;
                wts[p2] = cfg.weight;
                cursor[vv as usize] += 1;
            }
        }
    }

    let off_path = cfg.out_dir.join(format!("{}_csrOffset_u32.bin", cfg.graph_name));
    let dst_path = cfg.out_dir.join(format!("{}_csrDest_u32.bin", cfg.graph_name));
    let w_path = cfg.out_dir.join(format!("{}_csrWeight_u32.bin", cfg.graph_name));

    eprintln!("[WRITE] {:?}", off_path);
    write_u32_bin(&off_path, &offsets)?;
    eprintln!("[WRITE] {:?}", dst_path);
    write_u32_bin(&dst_path, &dest)?;
    eprintln!("[WRITE] {:?}", w_path);
    write_u32_bin(&w_path, &wts)?;

    eprintln!("[DONE] V={} E={} out_dir={:?}", v, e, cfg.out_dir);
    Ok(())
}
