use cust::prelude::*;
use crate::graph::subgraph::GpuSubgraph;
use cust::error::CudaResult;



pub struct GpuSubgraphDevice {
    pub offsets_d: DeviceBuffer<u32>,
    pub dst_d: DeviceBuffer<u32>,
    pub edge_cut_d: Option<DeviceBuffer<u32>>,
    pub cut_vertex: Option<u32>,
    pub m_gpu: usize,
}

impl GpuSubgraphDevice {
    pub fn from_host(sub: &GpuSubgraph) -> CudaResult<Self> {
        let offsets = sub.gpu_offsets();
        let dst = sub.gpu_dst();

        let offsets_d = DeviceBuffer::from_slice(offsets)?;
        let dst_d = DeviceBuffer::from_slice(dst)?;

        let edge_cut_d = match sub.gpu_edge_cut_opt() {
            Some(x) => Some(DeviceBuffer::from_slice(x)?),
            None => None,
        };

        Ok(Self {
            offsets_d,
            dst_d,
            edge_cut_d,
            cut_vertex: sub.cut_vertex_opt(),
            m_gpu: dst.len(),
        })
    }
}
