use cust::prelude::*;
use rust_het_graph::gpu::runtime::INC_PTX;

fn main() -> cust::error::CudaResult<()> {

    let _ctx = cust::quick_init()?;

    let module = Module::from_ptx(INC_PTX, &[])?;
    let func = module.get_function("inc_u32")?;

    let n: u32 = 1 << 20; 

    let h_in: Vec<u32> = (0..n).collect();
    let mut h_out: Vec<u32> = vec![0; n as usize];

    let d_in = DeviceBuffer::from_slice(&h_in)?;
    let mut d_out = DeviceBuffer::<u32>::zeroed(n as usize)?;

    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    let threads_per_block: u32 = 256;
    let blocks: u32 = (n + threads_per_block - 1) / threads_per_block;

    unsafe {
        launch!(
            func<<<blocks, threads_per_block, 0, stream>>>(
                d_in.as_device_ptr(),
                d_out.as_device_ptr(),
                n
            )
        )?;
    }

    stream.synchronize()?;
    d_out.copy_to(&mut h_out)?;

    for i in 0..(n as usize) {
        let expected = h_in[i] + 1;
        if h_out[i] != expected {
            panic!("Mismatch at i={i}: got {}, expected {}", h_out[i], expected);
        }
    }

    println!("GPU smoke test OK (n={n}, blocks={blocks}, tpb={threads_per_block})");
    Ok(())
}

