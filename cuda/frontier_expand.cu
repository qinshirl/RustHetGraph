#include <stdint.h>

// degrees for each vertex in the frontier
extern "C" __global__
void frontier_degrees_u32(
    const uint32_t* csr_offsets,      
    const uint32_t* frontier,         
    uint32_t* deg_out,                
    uint32_t frontier_len
) {
    uint32_t i = (uint32_t)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < frontier_len) {
        uint32_t u = frontier[i];
        deg_out[i] = csr_offsets[u + 1] - csr_offsets[u];
    }
}

// write neighbors (in CSR order) into candidates_out using out_offsets
extern "C" __global__
void frontier_write_neighbors_u32(
    const uint32_t* csr_offsets,     
    const uint32_t* csr_dst,         
    const uint32_t* frontier,         
    const uint32_t* out_offsets,     
    uint32_t* candidates_out,       
    uint32_t frontier_len
) {
    uint32_t i = (uint32_t)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < frontier_len) {
        uint32_t u = frontier[i];
        uint32_t start = csr_offsets[u];
        uint32_t end   = csr_offsets[u + 1];
        uint32_t out_base = out_offsets[i];

        for (uint32_t e = start; e < end; ++e) {
            candidates_out[out_base + (e - start)] = csr_dst[e];
        }
    }
}
// write neighbors with visited filter (sentinel for visited)
extern "C" __global__
void frontier_write_neighbors_filter_sentinel_u32(
    const uint32_t* csr_offsets,      
    const uint32_t* csr_dst,          
    const uint32_t* frontier,         
    const uint32_t* out_offsets,      
    const uint8_t* visited,           
    uint32_t* candidates_out,         
    uint32_t frontier_len
) {
    uint32_t i = (uint32_t)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < frontier_len) {
        uint32_t u = frontier[i];
        uint32_t start = csr_offsets[u];
        uint32_t end   = csr_offsets[u + 1];
        uint32_t out_base = out_offsets[i];

        for (uint32_t e = start; e < end; ++e) {
            uint32_t v = csr_dst[e];
            // Write sentinel for visited nodes (no compaction yet)
            candidates_out[out_base + (e - start)] = (visited[v] == 0) ? v : 0xFFFF'FFFFu;
        }
    }
}


// write neighbors with visited filter and atomic compaction        

extern "C" __global__
void frontier_write_neighbors_atomic_u32(
    const uint32_t* csr_offsets,      
    const uint32_t* csr_dst,          
    const uint32_t* frontier,         
    const uint8_t* visited,          
    uint32_t* candidates_out,         
    uint32_t* write_count,            
    uint32_t frontier_len
) {
    uint32_t i = (uint32_t)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < frontier_len) {
        uint32_t u = frontier[i];
        uint32_t start = csr_offsets[u];
        uint32_t end   = csr_offsets[u + 1];

        for (uint32_t e = start; e < end; ++e) {
            uint32_t v = csr_dst[e];
            if (visited[v] == 0) {
                // Atomic compaction into packed output
                uint32_t pos = atomicAdd(write_count, 1u);
                candidates_out[pos] = v;
            }
        }
    }
}

// write neighbors with atomic visited marking and compaction
extern "C" __global__
void frontier_write_neighbors_atomic_mark_u32(
    const uint32_t* csr_offsets,
    const uint32_t* csr_dst,
    const uint32_t* frontier,
    uint32_t* visited,            // u32 visited now
    uint32_t* candidates_out,
    uint32_t* write_count,
    uint32_t frontier_len
) {
    uint32_t i = (uint32_t)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < frontier_len) {
        uint32_t u = frontier[i];
        uint32_t start = csr_offsets[u];
        uint32_t end   = csr_offsets[u + 1];

        for (uint32_t e = start; e < end; ++e) {
            uint32_t v = csr_dst[e];

            //only first discoverer writes it
            if (atomicCAS((unsigned int*)&visited[v], 0u, 1u) == 0u) {
                uint32_t pos = atomicAdd(write_count, 1u);
                candidates_out[pos] = v;
            }
        }
    }
}

// write neighbors with atomic visited marking, distance update, and compaction 

extern "C" __global__
void frontier_write_neighbors_atomic_mark_dist_i32(
    const uint32_t* csr_offsets,     
    const uint32_t* csr_dst,        
    const uint32_t* frontier,       
    uint32_t* visited,               
    int32_t* dist,                   
    uint32_t* frontier_next,         
    uint32_t* write_count,           
    uint32_t frontier_len,
    int32_t level                    
) {
    uint32_t i = (uint32_t)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < frontier_len) {
        uint32_t u = frontier[i];
        uint32_t start = csr_offsets[u];
        uint32_t end   = csr_offsets[u + 1];

        for (uint32_t e = start; e < end; ++e) {
            uint32_t v = csr_dst[e];

            // claim v
            if (atomicCAS((unsigned int*)&visited[v], 0u, 1u) == 0u) {
                dist[v] = level + 1;
                uint32_t pos = atomicAdd(write_count, 1u);
                frontier_next[pos] = v;
            }
        }
    }
}

// subgraph extraction, write all neighbors of frontier into out (no filtering)
// SAFE: tid<k, u<gpu_n, clamp edges to gpu_dst_len, cap output by out_cap
extern "C" __global__
void frontier_expand_subgraph_only(
    const uint32_t* frontier,        // [k]
    uint32_t k,
    const uint32_t* gpu_offsets,     // [gpu_n + 1]
    const uint32_t* gpu_dst,         // [gpu_dst_len]
    uint32_t gpu_n,
    uint32_t gpu_dst_len,
    uint32_t* out,                   // [out_cap]
    uint32_t* write_count,           // single u32
    uint32_t out_cap
) {
    uint32_t tid = (uint32_t)(blockIdx.x * blockDim.x + threadIdx.x);
    if (tid >= k) return;

    uint32_t u = frontier[tid];
    if (u >= gpu_n) return;

    uint32_t start = gpu_offsets[u];
    uint32_t end   = gpu_offsets[u + 1];

    // Defensive clamp (protect against bad offsets)
    if (start > gpu_dst_len) return;
    if (end > gpu_dst_len) end = gpu_dst_len;

    for (uint32_t e = start; e < end; ++e) {
        uint32_t v = gpu_dst[e];
        uint32_t pos = atomicAdd(write_count, 1u);
        if (pos < out_cap) {
            out[pos] = v;
        }
    }
}

