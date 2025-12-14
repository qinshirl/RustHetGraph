use std::{fs::File, io::{Read, Write}, path::Path};

use super::csr::CsrGraph;

pub fn read_u32_bin(path: &Path) -> Result<Vec<u32>, String> {
    let mut f = File::open(path).map_err(|e| format!("open {:?}: {e}", path))?;
    let mut buf = Vec::new();
    f.read_to_end(&mut buf).map_err(|e| format!("read {:?}: {e}", path))?;

    if buf.len() % 4 != 0 {
        return Err(format!("file {:?} length {} not divisible by 4", path, buf.len()));
    }

    let mut out = Vec::with_capacity(buf.len() / 4);
    for chunk in buf.chunks_exact(4) {
        out.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(out)
}

pub fn write_u32_bin(path: &Path, data: &[u32]) -> Result<(), String> {
    let mut f = File::create(path).map_err(|e| format!("create {:?}: {e}", path))?;

    // Write little-endian u32, exactly like CGgraph expects on x86.
    // This is deterministic and matches your read_u32_bin().
    let mut buf = Vec::with_capacity(data.len() * 4);
    for &x in data {
        buf.extend_from_slice(&x.to_le_bytes());
    }

    f.write_all(&buf).map_err(|e| format!("write {:?}: {e}", path))?;
    Ok(())
}

pub fn load_csr_from_dir(dir: impl AsRef<Path>) -> Result<CsrGraph, String> {
    let dir = dir.as_ref();

    // Match your current filenames exactly
    let offset_path = dir.join("native_csrOffset_u32.bin");
    let dest_path   = dir.join("native_csrDest_u32.bin");
    let weight_path = dir.join("native_csrWeight_u32.bin");

    let offsets = read_u32_bin(&offset_path)?;
    let dst = read_u32_bin(&dest_path)?;

    let w = if weight_path.exists() {
        Some(read_u32_bin(&weight_path)?)
    } else {
        None
    };

    Ok(CsrGraph { offsets, dst, w })
}


pub fn load_reordered_csr_from_dir(dir: impl AsRef<Path>) -> Result<CsrGraph, String> {
    let dir = dir.as_ref();

    let offset_path = dir.join("cggraphRV1_5_csrOffset_u32.bin");
    let dest_path   = dir.join("cggraphRV1_5_csrDest_u32.bin");
    let weight_path = dir.join("cggraphRV1_5_csrWeight_u32.bin");

    let offsets = read_u32_bin(&offset_path)?;
    let dst = read_u32_bin(&dest_path)?;

    let w = if weight_path.exists() {
        Some(read_u32_bin(&weight_path)?)
    } else {
        None
    };

    Ok(CsrGraph { offsets, dst, w })
}


pub fn write_i32_bin(path: &Path, data: &[i32]) -> Result<(), String> {
    let mut f = File::create(path).map_err(|e| format!("create {:?}: {e}", path))?;

    let mut buf = Vec::with_capacity(data.len() * 4);
    for &x in data {
        buf.extend_from_slice(&x.to_le_bytes());
    }
    f.write_all(&buf).map_err(|e| format!("write {:?}: {e}", path))?;
    Ok(())
}

pub fn write_u64_bin(path: &Path, data: &[u64]) -> Result<(), String> {
    let mut f = File::create(path).map_err(|e| format!("create {:?}: {e}", path))?;

    let mut buf = Vec::with_capacity(data.len() * 8);
    for &x in data {
        buf.extend_from_slice(&x.to_le_bytes());
    }
    f.write_all(&buf).map_err(|e| format!("write {:?}: {e}", path))?;
    Ok(())
}
