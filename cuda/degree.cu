#include <stdint.h>

extern "C" __global__
void csr_degree_u32(const uint32_t* offsets, uint32_t* deg, uint32_t n) {
    uint32_t v = (uint32_t)(blockIdx.x * blockDim.x + threadIdx.x);
    if (v < n) {
        deg[v] = offsets[v + 1] - offsets[v];
    }
}
