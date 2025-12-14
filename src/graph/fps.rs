use std::fs::File;
use std::io::{Read, BufReader};
use std::path::Path;

//3x u32 + padding + 2x f64 = 32 bytes 
#[derive(Clone, Copy, Debug)]
pub struct SpeedRecord {
    pub ite_id: u32,
    pub active_v: u32,
    pub active_e: u32,
    pub time_ms: f64,
    pub total_time_ms: f64,
}

fn read_u64_le<R: Read>(r: &mut R) -> Result<u64, String> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b).map_err(|e| format!("read u64: {e}"))?;
    Ok(u64::from_le_bytes(b))
}

fn read_u32_le<R: Read>(r: &mut R) -> Result<u32, String> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b).map_err(|e| format!("read u32: {e}"))?;
    Ok(u32::from_le_bytes(b))
}

fn read_f64_le<R: Read>(r: &mut R) -> Result<f64, String> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b).map_err(|e| format!("read f64: {e}"))?;
    Ok(f64::from_le_bytes(b))
}

// Read a CGgraph speed file

// size_t outerSize
// repeat outerSize times:
// size_t innerSize
// innerSize * SpeedRecord_type bytes
pub fn read_speed_records<P: AsRef<Path>>(path: P) -> Result<Vec<Vec<SpeedRecord>>, String> {
    let f = File::open(&path).map_err(|e| format!("open {:?}: {e}", path.as_ref()))?;
    let mut r = BufReader::new(f);

    let outer = read_u64_le(&mut r)? as usize;
    let mut runs: Vec<Vec<SpeedRecord>> = Vec::with_capacity(outer);

    for run_id in 0..outer {
        let inner = read_u64_le(&mut r)? as usize;
        let mut recs: Vec<SpeedRecord> = Vec::with_capacity(inner);

        for _ in 0..inner {
            let ite_id = read_u32_le(&mut r)?;
            let active_v = read_u32_le(&mut r)?;
            let active_e = read_u32_le(&mut r)?;

            let mut pad = [0u8; 4];
            r.read_exact(&mut pad).map_err(|e| format!("read padding: {e}"))?;

            let time_ms = read_f64_le(&mut r)?;
            let total_time_ms = read_f64_le(&mut r)?;

            recs.push(SpeedRecord {
                ite_id,
                active_v,
                active_e,
                time_ms,
                total_time_ms,
            });
        }

        if !recs.is_empty() && recs[0].ite_id != 1 {
            eprintln!(
                "[fps] warning: run {run_id} first ite_id={}, expected 1",
                recs[0].ite_id
            );
        }

        runs.push(recs);
    }

    Ok(runs)
}


pub fn print_speed_summary(tag: &str, runs: &[Vec<SpeedRecord>], k: usize) {
    println!("[D4] {tag}: outer runs = {}", runs.len());
    for (ri, run) in runs.iter().enumerate() {
        println!(
            "[D4] {tag}[run {ri}] inner records = {}, last total_time_ms = {:.3}",
            run.len(),
            run.last().map(|x| x.total_time_ms).unwrap_or(0.0)
        );

        for i in 0..k.min(run.len()) {
            let r = run[i];
            println!(
                "  {tag}[run {ri}][{i:02}] ite={} activeV={} activeE={} time_ms={:.6} total_ms={:.6}",
                r.ite_id, r.active_v, r.active_e, r.time_ms, r.total_time_ms
            );
        }
    }
}
pub fn run_time_sanity(run: &[SpeedRecord]) -> (f64, f64) {
    let sum_time_ms: f64 = run.iter().map(|r| r.time_ms).sum();
    let stored_total_ms: f64 = run.last().map(|r| r.total_time_ms).unwrap_or(0.0);

    (sum_time_ms, stored_total_ms)
}
