#[derive(Debug)]
pub struct CsrGraph {
    pub offsets: Vec<u32>,          // len=n+1
    pub dst: Vec<u32>,              // len=m
    pub w: Option<Vec<u32>>,        // len=m 
}

impl CsrGraph {
    pub fn n(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }
    pub fn m(&self) -> usize {
        self.dst.len()
    }
    pub fn neighbors(&self, u: u32) -> &[u32] {
        let u = u as usize;
        let start = self.offsets[u] as usize;
        let end = self.offsets[u + 1] as usize;
        &self.dst[start..end]
    }
}
