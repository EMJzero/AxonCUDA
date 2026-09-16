#include <tuple>

#include <cub/cub.cuh>

#include "thruster.cuh"

#include "hgraph.hpp"
#include "runconfig_plc.hpp"

#include "prep.cuh"

#include "utils.cuh"
#include "defines_plc.cuh"

using namespace config_plc;

std::tuple<uint32_t*, dim_t*> buildTouchingHost(
    const runconfig &cfg,
    const HyperGraph& hg
) {
    ERR(cfg) std::cerr << "WARNING: moving incidence sets host -> device will take a while...\n";

    // HP: hedges already internally deduplicated (acyclic), keeping the dst whenever a duplicate is between srcs and dsts
    uint32_t *d_touching = nullptr;
    dim_t *d_touching_offsets = nullptr;

    const uint32_t num_nodes = hg.nodes();

    std::vector<uint32_t> touching_hedges;
    std::vector<dim_t> touching_hedges_offsets;
    touching_hedges.reserve(hg.hedgesFlat().size());
    touching_hedges_offsets.reserve(num_nodes + 1);

    // prepare touching sets
    for (uint32_t n = 0; n < hg.nodes(); ++n) {
        touching_hedges_offsets.push_back(touching_hedges.size());
        // NOTE: must put in inbounds first!
        for (uint32_t h : hg.inboundSortedIds(n))
            touching_hedges.push_back(h);
        for (uint32_t h : hg.outboundSortedIds(n))
            touching_hedges.push_back(h);
    }
    touching_hedges_offsets.push_back(touching_hedges.size());
    dim_t touching_hedges_size = touching_hedges.size();

    CUDA_CHECK(cudaMalloc(&d_touching, touching_hedges_size * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_touching_offsets, (num_nodes + 1) * sizeof(dim_t)));

    CUDA_CHECK(cudaMemcpy(d_touching, touching_hedges.data(), touching_hedges_size * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_touching_offsets, touching_hedges_offsets.data(), (num_nodes + 1) * sizeof(dim_t), cudaMemcpyHostToDevice));

    return std::make_tuple(d_touching, d_touching_offsets);
}

std::tuple<uint32_t*, dim_t*> buildTouching(
    const runconfig &cfg,
    const uint32_t *d_hedges,
    const dim_t *d_hedges_offsets,
    const uint32_t num_nodes,
    const uint32_t num_hedges
) {
    // HP: hedges already internally deduplicated (acyclic), keeping the dst whenever a duplicate is between srcs and dsts
    uint32_t *d_touching = nullptr;
    dim_t *d_touching_offsets = nullptr;

    CUDA_CHECK(cudaMalloc(&d_touching_offsets, (num_nodes + 1) * sizeof(dim_t)));
    CUDA_CHECK(cudaMemset(d_touching_offsets, 0x00, (num_nodes + 1) * sizeof(dim_t))); // remember to leave the first offset at 0
    
    {
        // launch configuration - touching count
        int threads_per_block = 128; // 128/32 -> 4 warps per block
        int warps_per_block = threads_per_block / WARP_SIZE;
        int num_warps_needed = num_hedges; // 1 warp per hedge
        int blocks = (num_warps_needed + warps_per_block - 1) / warps_per_block;
        // launch - touching count
        LAUNCH(cfg) RUN << "touching count kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
        touching_count_kernel<<<blocks, threads_per_block>>>(
            d_hedges,
            d_hedges_offsets,
            num_hedges,
            d_touching_offsets
        );
        DBG(cfg) CUDA_CHECK(cudaGetLastError());
        DBG(cfg) CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    thrust::device_ptr<dim_t> t_touching_offsets(d_touching_offsets);
    thrust::inclusive_scan(t_touching_offsets, t_touching_offsets + (num_nodes + 1), t_touching_offsets); // in-place exclusive scan (the last element is set to zero and thus collects the full reduce)
    dim_t touching_size = 0; // last value in the inclusive scan = full reduce = total number of touching hedges among all sets
    CUDA_CHECK(cudaMemcpy(&touching_size, d_touching_offsets + num_nodes, sizeof(dim_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMalloc(&d_touching, touching_size * sizeof(uint32_t)));
    
    uint32_t *d_inserted_count = nullptr;
    CUDA_CHECK(cudaMalloc(&d_inserted_count, num_nodes * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_inserted_count, 0x00, num_nodes * sizeof(uint32_t)));
    {
        // launch configuration - touching build kernel
        int threads_per_block = 128; // 128/32 -> 4 warps per block
        int warps_per_block = threads_per_block / WARP_SIZE;
        int num_warps_needed = num_hedges; // 1 warp per hedge
        int blocks = (num_warps_needed + warps_per_block - 1) / warps_per_block;
        // launch - touching build kernel
        LAUNCH(cfg) RUN << "touching build kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
        touching_build_kernel<<<blocks, threads_per_block>>>(
            d_hedges,
            d_hedges_offsets,
            d_touching_offsets,
            num_hedges,
            d_touching,
            d_inserted_count
        );
        DBG(cfg) CUDA_CHECK(cudaGetLastError());
        DBG(cfg) CUDA_CHECK(cudaDeviceSynchronize());
    }
    CUDA_CHECK(cudaFree(d_inserted_count));

    // each node's hedges were claimed by racing atomics, so sort every set to pin down its order
    // => that order decides which lane of a warp visits which pin later on, and with it the summation order of every touching-based float reduction
    uint32_t *d_touching_buffer = nullptr; // CUB segmented sort buffer
    CUDA_CHECK(cudaMalloc(&d_touching_buffer, touching_size * sizeof(uint32_t)));
    cub::DoubleBuffer<uint32_t> c_touching_double_buffer(d_touching, d_touching_buffer);
    void* c_touching_storage = nullptr;
    size_t c_touching_storage_bytes = 0;
    cub::DeviceSegmentedSort::SortKeys(
        c_touching_storage, c_touching_storage_bytes, c_touching_double_buffer,
        touching_size, num_nodes, d_touching_offsets, d_touching_offsets + 1
    );
    CUDA_CHECK(cudaMalloc(&c_touching_storage, c_touching_storage_bytes));
    cub::DeviceSegmentedSort::SortKeys(
        c_touching_storage, c_touching_storage_bytes, c_touching_double_buffer,
        touching_size, num_nodes, d_touching_offsets, d_touching_offsets + 1
    );
    DBG(cfg) CUDA_CHECK(cudaDeviceSynchronize());
    if (c_touching_double_buffer.Current() != d_touching) {
        uint32_t* tmp = d_touching_buffer;
        d_touching_buffer = d_touching;
        d_touching = tmp;
    }
    CUDA_CHECK(cudaFree(d_touching_buffer));
    CUDA_CHECK(cudaFree(c_touching_storage));

    return std::make_tuple(d_touching, d_touching_offsets);
}