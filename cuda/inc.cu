#include <stdint.h>

extern "C" __global__
void inc_u32(const uint32_t* in, uint32_t* out, uint32_t n) {
    uint32_t tid = (uint32_t)(blockIdx.x * blockDim.x + threadIdx.x);
    if (tid < n) {
        out[tid] = in[tid] + 1u;
    }
}
