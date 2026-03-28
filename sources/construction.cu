#include <tuple>

#include "thruster.cuh"

#include <cub/cub.cuh>

#include "runconfig.hpp"

#include "utils.cuh"
#include "defines.cuh"
#include "construction.cuh"

std::tuple<dim_t, uint32_t*, dim_t*> buildNeighbors(
    const runconfig cfg,
    const uint32_t *d_hedges,
    const dim_t *d_hedges_offsets,
    const uint32_t *d_touching,
    const dim_t *d_touching_offsets,
    const uint32_t num_nodes,
    const uint32_t max_neighbors,
    uint32_t *d_neighbors,
    dim_t *d_neighbors_offsets
) {
    // HP: no duplicates in neighbors, no one's own self among one's neighbors
    // uses a two-step method, first just counting, then writing, to allocate exactly the amount of memory needed, since neighborhoods can explode quickly...
    // if there is enough memory, a speedier version is used, that replaced the scatter with a direct pack from the initial oversized allocation!
    uint32_t *d_oversized_neighbors = nullptr;
    dim_t init_max_neighbors = (dim_t)std::ceil(cfg.oversized_multiplier * (float)max_neighbors);
    
    size_t free_bytes, total_bytes;
    cudaMemGetInfo(&free_bytes, &total_bytes);
    // check if there could be space to allocate both oversized neighbors and final neighbors at once; with no better guess, use 'max_neighbors' to estimate the final neighbors size...
    bool direct_scatter_neighbors = (num_nodes * init_max_neighbors /*oversized*/ + num_nodes * max_neighbors /*final upper bound*/) * sizeof(uint32_t) + num_nodes * sizeof(dim_t) /*offsets*/ < free_bytes;
    // no pack? can spare space in the oversized buffer equal to the amount of shared memory used for fast deduping
    if (!direct_scatter_neighbors) init_max_neighbors = init_max_neighbors > SM_MAX_BLOCK_DEDUPE_BUFFER_SIZE ? init_max_neighbors - SM_MAX_BLOCK_DEDUPE_BUFFER_SIZE : 0;
    init_max_neighbors = max(init_max_neighbors, (dim_t)GM_MIN_BLOCK_DEDUPE_BUFFER_SIZE);
    
    if (num_nodes * init_max_neighbors * sizeof(uint32_t) > (1ull << 32))
        std::cout << "Allocating " << std::fixed << std::setprecision(1) << (float)(num_nodes * init_max_neighbors * sizeof(uint32_t)) / (1 << 30) << " GB for neighbors deduplication ...\n";
    CUDA_CHECK(cudaMalloc(&d_oversized_neighbors, num_nodes * init_max_neighbors * sizeof(uint32_t))); // space for spilling deduplication hash-sets
    CUDA_CHECK(cudaMalloc(&d_neighbors_offsets, (num_nodes + 1) * sizeof(dim_t))); // node -> neighbors set start idx in d_neighbors
    thrust::device_ptr<dim_t> t_neigh_offsets(d_neighbors_offsets);
    {
        // launch configuration - neighborhoods count kernel
        int blocks = num_nodes;
        int threads_per_block = 256; // 256/32 -> 8 warps per block
        std::cout << "Running neighborhoods count kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
        // launch - neighborhoods count kernel
        neighborhoods_count_kernel<<<blocks, threads_per_block>>>(
            d_hedges,
            d_hedges_offsets,
            d_touching,
            d_touching_offsets,
            num_nodes,
            init_max_neighbors,
            direct_scatter_neighbors,
            d_oversized_neighbors,
            d_neighbors_offsets
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    if (!direct_scatter_neighbors) CUDA_CHECK(cudaFree(d_oversized_neighbors)); // no pack? free oversized immediately
    
    // correct the max neighbors count estimate
    auto actual_max_neighbors = thrust::max_element(t_neigh_offsets, t_neigh_offsets + num_nodes);
    dim_t actual_max_neighbors_offset = static_cast<dim_t>(actual_max_neighbors - t_neigh_offsets);
    dim_t new_max_neighbors;
    CUDA_CHECK(cudaMemcpy(&new_max_neighbors, d_neighbors_offsets + actual_max_neighbors_offset, sizeof(dim_t), cudaMemcpyDeviceToHost));
    std::cout << "Max neighbors estimate corrected to " << new_max_neighbors << "\n";
    
    // compute final offsets
    thrust::exclusive_scan(t_neigh_offsets, t_neigh_offsets + (num_nodes + 1), t_neigh_offsets); // in-place exclusive scan (the last element is set to zero and thus collects the full reduce)
    dim_t total_neighbors;
    CUDA_CHECK(cudaMemcpy(&total_neighbors, d_neighbors_offsets + num_nodes, sizeof(dim_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMalloc(&d_neighbors, total_neighbors * sizeof(uint32_t)));
    if (direct_scatter_neighbors) {
        // pack oversized neighbors in their final tight-fit subarrays
        // launch configuration - neighborhoods pack kernel
        int threads_per_block = 128; // 128/32 -> 4 warps per block
        int warps_per_block = threads_per_block / WARP_SIZE;
        int num_warps_needed = num_nodes ; // 1 warp per node
        int blocks = (num_warps_needed + warps_per_block - 1) / warps_per_block;
        std::cout << "Running neighborhoods pack kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
        // launch - neighborhoods scatter kernel
        pack_segments<<<blocks, threads_per_block>>>(
            d_oversized_neighbors,
            d_neighbors_offsets,
            num_nodes,
            init_max_neighbors,
            d_neighbors
        );
    } else {
        // write neighbors at their correct offset
        // launch configuration - neighborhoods scatter kernel
        int blocks = num_nodes;
        int threads_per_block = 256; // 256/32 -> 8 warps per block
        std::cout << "Running neighborhoods scatter kernel (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
        // launch - neighborhoods scatter kernel
        neighborhoods_scatter_kernel<<<blocks, threads_per_block>>>(
            d_hedges,
            d_hedges_offsets,
            d_touching,
            d_touching_offsets,
            num_nodes,
            d_neighbors_offsets,
            d_neighbors
        );
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    if (direct_scatter_neighbors) CUDA_CHECK(cudaFree(d_oversized_neighbors)); // pack? free oversized afterwards

    return std::make_tuple(new_max_neighbors, d_neighbors, d_neighbors_offsets);
}

std::tuple<dim_t, uint32_t*, dim_t*> coarsenNeighbors(
    const runconfig cfg,
    const uint32_t *d_groups,
    const uint32_t *d_ungroups,
    const dim_t *d_ungroups_offsets,
    const uint32_t curr_num_nodes,
    const uint32_t new_num_nodes,
    const uint32_t max_neighbors,
    uint32_t *d_neighbors,
    dim_t *d_neighbors_offsets
) {
    // prepare coarse neighbors buffers
    uint32_t *d_coarse_neighbors = nullptr;
    uint32_t *d_coarse_oversized_neighbors = nullptr;
    dim_t *d_coarse_neighbors_offsets = nullptr;
    dim_t curr_max_neighbors = (dim_t)(cfg.oversized_multiplier * (float)max_neighbors); // add a bit of safety-room to compensate for the flat scaling by 'new_num_nodes / curr_num_nodes'
    
    // if there is enough memory for the full oversized buffer, SM dischard included, a speedier version is used, that replaced the scatter with a direct pack from the initial oversized allocation!
    size_t free_bytes, total_bytes;
    cudaMemGetInfo(&free_bytes, &total_bytes);
    // NOTE: no need to check if there could be space to allocate both oversized neighbors and final neighbors at once, if the oversized fits, then the new neighbors are allocated either after the oversized is freed, or after the original neighbors are freed
    bool direct_scatter_coarse_neighbors = (curr_num_nodes * curr_max_neighbors /*oversized (SM included)*/ + new_num_nodes * max_neighbors /*final upper bound*/) * sizeof(uint32_t) + new_num_nodes * sizeof(dim_t) /*offsets*/ < free_bytes;
    // no pack? can spare space in the oversized buffer equal to the amount of shared memory used for fast deduping
    if (!direct_scatter_coarse_neighbors) curr_max_neighbors = curr_max_neighbors > MAX_SM_WARP_DEDUPE_BUFFER_SIZE ? curr_max_neighbors - MAX_SM_WARP_DEDUPE_BUFFER_SIZE : 0; // save the spaced for the duplicates caught in SM
    curr_max_neighbors = max(curr_max_neighbors, (dim_t)MIN_GM_WARP_DEDUPE_BUFFER_SIZE); // just some ensurance...
    
    if (curr_num_nodes * curr_max_neighbors * sizeof(uint32_t) > (1ull << 32))
        std::cout << "Allocating " << std::fixed << std::setprecision(1) << (float)(curr_num_nodes * curr_max_neighbors * sizeof(uint32_t)) / (1 << 30) << " GB for neighbors deduplication ...\n";
    CUDA_CHECK(cudaMalloc(&d_coarse_oversized_neighbors, curr_num_nodes * curr_max_neighbors * sizeof(uint32_t))); // space for spilling deduplication hash-sets
    CUDA_CHECK(cudaMalloc(&d_coarse_neighbors_offsets, (1 + new_num_nodes) * sizeof(dim_t))); // NOTE: the number nodes decreases!
    CUDA_CHECK(cudaMemset(d_coarse_neighbors_offsets, 0x00, sizeof(dim_t))); // init. the first offset at 0
    thrust::device_ptr<dim_t> t_coarse_neigh_offsets(d_coarse_neighbors_offsets);
    {
        // launch configuration - coarsening kernel (neighbors - count)
        int threads_per_block = 128; // 128/32 -> 4 warps per block
        int warps_per_block = threads_per_block / WARP_SIZE;
        int num_warps_needed = new_num_nodes ; // 1 warp per group
        int blocks = (num_warps_needed + warps_per_block - 1) / warps_per_block;
        // compute shared memory per block (bytes)
        size_t bytes_per_warp = MAX_SM_WARP_DEDUPE_BUFFER_SIZE * sizeof(uint32_t);
        size_t shared_bytes = warps_per_block * bytes_per_warp;
        // launch - coarsening kernel (neighbors - count)
        std::cout << "Running coarsening kernel (neighbors - count) (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
        apply_coarsening_neighbors_count<<<blocks, threads_per_block, shared_bytes>>>(
            d_neighbors,
            d_neighbors_offsets,
            d_groups,
            d_ungroups,
            d_ungroups_offsets,
            new_num_nodes,
            curr_max_neighbors,
            direct_scatter_coarse_neighbors,
            d_coarse_oversized_neighbors,
            d_coarse_neighbors_offsets
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    if (!direct_scatter_coarse_neighbors) CUDA_CHECK(cudaFree(d_coarse_oversized_neighbors));
    if (direct_scatter_coarse_neighbors) { CUDA_CHECK(cudaFree(d_neighbors)); CUDA_CHECK(cudaFree(d_neighbors_offsets)); }
    
    // correct the max neighbors count estimate
    auto actual_coarse_max_neighbors = thrust::max_element(t_coarse_neigh_offsets, t_coarse_neigh_offsets + new_num_nodes + 1);
    dim_t actual_coarse_max_neighbors_offset = static_cast<dim_t>(actual_coarse_max_neighbors - t_coarse_neigh_offsets);
    dim_t new_max_neighbors;
    CUDA_CHECK(cudaMemcpy(&new_max_neighbors, d_coarse_neighbors_offsets + actual_coarse_max_neighbors_offset, sizeof(dim_t), cudaMemcpyDeviceToHost));
    std::cout << "Max neighbors estimate corrected to " << new_max_neighbors << "\n";
    
    // compute final offsets
    thrust::inclusive_scan(t_coarse_neigh_offsets, t_coarse_neigh_offsets + (new_num_nodes + 1), t_coarse_neigh_offsets); // in-place exclusive scan (the last element is set to zero and thus collects the full reduce)
    dim_t new_neighbors_size = 0; // last value in the inclusive scan = full reduce = total number of neighbors among all sets
    CUDA_CHECK(cudaMemcpy(&new_neighbors_size, d_coarse_neighbors_offsets + new_num_nodes, sizeof(dim_t), cudaMemcpyDeviceToHost));
    // NOTE: rebuilding neighbors from scratch makes no sense: if the "oversized" buffer could fit, no reason the new neighbors shouldn't!
    // this alloc should never fail, since it occurs in the space left by either the oversized buffer or previous neighbors
    CUDA_CHECK(cudaMalloc(&d_coarse_neighbors, new_neighbors_size * sizeof(uint32_t)));
    if (direct_scatter_coarse_neighbors) {
        // pack oversized coarse neighbors in their final tight-fit subarrays
        // launch configuration - coarsening kernel (neighbors - pack)
        int threads_per_block = 128; // 128/32 -> 4 warps per block
        int warps_per_block = threads_per_block / WARP_SIZE;
        int num_warps_needed = new_num_nodes; // 1 warp per group
        int blocks = (num_warps_needed + warps_per_block - 1) / warps_per_block;
        std::cout << "Running coarsening kernel (neighbors - pack) (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
        // launch - neighborhoods scatter kernel
        pack_segments_varsize<<<blocks, threads_per_block>>>(
            d_coarse_oversized_neighbors,
            d_ungroups_offsets, // once more, ungroup offsets provide the offsets for the oversized buffer too
            d_coarse_neighbors_offsets,
            new_num_nodes,
            curr_max_neighbors,
            d_coarse_neighbors
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaFree(d_coarse_oversized_neighbors));
    } else {
        // launch configuration - coarsening kernel (neighbors - scatter)
        int threads_per_block = 128; // 128/32 -> 4 warps per block
        int warps_per_block = threads_per_block / WARP_SIZE;
        int num_warps_needed = new_num_nodes ; // 1 warp per group
        int blocks = (num_warps_needed + warps_per_block - 1) / warps_per_block;
        // compute shared memory per block (bytes)
        size_t bytes_per_warp = MAX_SM_WARP_DEDUPE_BUFFER_SIZE * sizeof(uint32_t);
        size_t shared_bytes = warps_per_block * bytes_per_warp;
        // launch - coarsening kernel (neighbors - scatter)
        std::cout << "Running coarsening kernel (neighbors - scatter) (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
        apply_coarsening_neighbors_scatter<<<blocks, threads_per_block, shared_bytes>>>(
            d_neighbors,
            d_neighbors_offsets,
            d_groups,
            d_ungroups,
            d_ungroups_offsets,
            new_num_nodes,
            d_coarse_neighbors_offsets,
            d_coarse_neighbors
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        // de-allocate old neighbors and replace them with coarse ones
        CUDA_CHECK(cudaFree(d_neighbors));
        CUDA_CHECK(cudaFree(d_neighbors_offsets));
    }

    return std::make_tuple(new_max_neighbors, d_coarse_neighbors, d_coarse_neighbors_offsets);
}

std::tuple<dim_t, dim_t, uint32_t*, dim_t*, uint32_t*> coarsenHedges(
    const runconfig cfg,
    const uint32_t *d_hedges,
    const dim_t *d_hedges_offsets,
    const uint32_t *d_srcs_count,
    const uint32_t *d_groups,
    const uint32_t num_hedges,
    const uint32_t max_hedge_size
) {
    uint32_t *d_coarse_hedges = nullptr;
    uint32_t *d_coarse_hedges_buffer = nullptr;
    uint32_t *d_coarse_oversized_hedges = nullptr;
    dim_t *d_coarse_hedges_offsets = nullptr;
    uint32_t* d_coarse_srcs_count = nullptr;
    
    dim_t curr_max_hedge_size = (dim_t)(cfg.oversized_multiplier * (float)max_hedge_size);
    curr_max_hedge_size = curr_max_hedge_size > MAX_SM_WARP_DEDUPE_BUFFER_SIZE ? curr_max_hedge_size - MAX_SM_WARP_DEDUPE_BUFFER_SIZE : 0;
    curr_max_hedge_size = max(curr_max_hedge_size, (dim_t)MIN_GM_WARP_DEDUPE_BUFFER_SIZE);
    
    if (num_hedges * curr_max_hedge_size * sizeof(uint32_t) > (1ull << 32))
        std::cout << "Allocating " << std::fixed << std::setprecision(1) << (float)(num_hedges * curr_max_hedge_size * sizeof(uint32_t)) / (1 << 30) << " GB for hedges deduplication ...\n";
    CUDA_CHECK(cudaMalloc(&d_coarse_oversized_hedges, num_hedges * curr_max_hedge_size * sizeof(uint32_t))); // space for spilling deduplication hash-sets
    CUDA_CHECK(cudaMalloc(&d_coarse_hedges_offsets, (1 + num_hedges) * sizeof(dim_t))); // NOTE: the number of hedges never decreases (for now), unlike that of nodes!
    CUDA_CHECK(cudaMalloc(&d_coarse_srcs_count, num_hedges * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_coarse_hedges_offsets, 0x00, sizeof(dim_t))); // init. the first offset at 0
    thrust::device_ptr<dim_t> t_coarse_hedges_offsets(d_coarse_hedges_offsets);
    
    // launch configuration - coarsening kernel (hedges - count)
    int threads_per_block = 128; // 128/32 -> 4 warps per block
    int warps_per_block = threads_per_block / WARP_SIZE;
    int num_warps_needed = num_hedges; // 1 warp per hedge
    int blocks = (num_warps_needed + warps_per_block - 1) / warps_per_block;
    // compute shared memory per block (bytes)
    size_t bytes_per_warp = MAX_SM_WARP_DEDUPE_BUFFER_SIZE * sizeof(uint32_t);
    size_t shared_bytes = warps_per_block * bytes_per_warp;
    // launch - coarsening kernel (hedges - count)
    std::cout << "Running coarsening kernel (hedges - count) (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
    apply_coarsening_hedges_count<<<blocks, threads_per_block, shared_bytes>>>(
        d_hedges,
        d_hedges_offsets,
        d_srcs_count,
        d_groups,
        num_hedges,
        curr_max_hedge_size,
        d_coarse_oversized_hedges,
        d_coarse_hedges_offsets,
        d_coarse_srcs_count
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(d_coarse_oversized_hedges));

    // correct the max hedge size estimate
    auto actual_coarse_max_hedge_size = thrust::max_element(t_coarse_hedges_offsets, t_coarse_hedges_offsets + num_hedges + 1);
    dim_t actual_coarse_max_hedge_size_offset = static_cast<dim_t>(actual_coarse_max_hedge_size - t_coarse_hedges_offsets);
    dim_t new_max_hedge_size;
    CUDA_CHECK(cudaMemcpy(&new_max_hedge_size, d_coarse_hedges_offsets + actual_coarse_max_hedge_size_offset, sizeof(dim_t), cudaMemcpyDeviceToHost));
    std::cout << "Max hedges estimate corrected to " << max_hedge_size << "\n";
    
    // NOTE: the scan wants the last index EXCLUDED, while the memcopy wants the last index exactly! That's why we use here the +1, and not later!
    thrust::inclusive_scan(t_coarse_hedges_offsets, t_coarse_hedges_offsets + (num_hedges + 1), t_coarse_hedges_offsets); // in-place exclusive scan (the last element collects the full reduce)
    dim_t new_hedges_size = 0; // last value in the inclusive scan = full reduce = total number of pins among all hedges
    CUDA_CHECK(cudaMemcpy(&new_hedges_size, d_coarse_hedges_offsets + num_hedges, sizeof(dim_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMalloc(&d_coarse_hedges, new_hedges_size * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_coarse_hedges_buffer, new_hedges_size * sizeof(uint32_t)));
    // launch configuration - coarsening kernel (hedges - scatter - dsts) - same as coarsening kernel (hedges - count)
    // launch - coarsening kernel (hedges - scatter - dsts)
    std::cout << "Running coarsening kernel (hedges - scatter - dsts) (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
    apply_coarsening_hedges_scatter_dsts<<<blocks, threads_per_block, shared_bytes>>>(
        d_hedges,
        d_hedges_offsets,
        d_srcs_count,
        d_groups,
        num_hedges,
        d_coarse_hedges_offsets,
        d_coarse_hedges
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // sort destinations (descending)
    cub::DoubleBuffer<uint32_t> c_coarse_hedges_double_buffer(d_coarse_hedges, d_coarse_hedges_buffer);
    void* c_hedges_storage = nullptr;
    size_t c_hedges_storage_bytes = 0;
    cub::DeviceSegmentedRadixSort::SortKeysDescending(c_hedges_storage, c_hedges_storage_bytes, c_coarse_hedges_double_buffer, new_hedges_size, num_hedges, d_coarse_hedges_offsets, d_coarse_hedges_offsets + 1, /*begin_bit=*/0, /*end_bit=*/sizeof(uint32_t) * 8, /*stream*/0);
    std::cout << "CUB segmented sort requiring " << std::fixed << std::setprecision(3) << (float)(new_hedges_size * sizeof(uint32_t)) / (1 << 30) << " GB of pong-buffer and " << std::fixed << std::setprecision(3) << ((float)c_hedges_storage_bytes) / (1 << 20) << " MB of temporary storage ...\n";
    cudaMalloc(&c_hedges_storage, c_hedges_storage_bytes);
    cub::DeviceSegmentedRadixSort::SortKeysDescending(c_hedges_storage, c_hedges_storage_bytes, c_coarse_hedges_double_buffer, new_hedges_size, num_hedges, d_coarse_hedges_offsets, d_coarse_hedges_offsets + 1, /*begin_bit=*/0, /*end_bit=*/sizeof(uint32_t) * 8, /*stream*/0);
    if (c_coarse_hedges_double_buffer.Current() != d_coarse_hedges) {
        uint32_t* tmp = d_coarse_hedges_buffer;
        d_coarse_hedges_buffer = d_coarse_hedges;
        d_coarse_hedges = tmp;
    }

    // launch configuration - coarsening kernel (hedges - scatter - srcs) - same as coarsening kernel (hedges - count)
    // launch - coarsening kernel (hedges - scatter - srcs)
    std::cout << "Running coarsening kernel (hedges - scatter - srcs) (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
    apply_coarsening_hedges_scatter_srcs<<<blocks, threads_per_block, shared_bytes>>>(
        d_hedges,
        d_hedges_offsets,
        d_srcs_count,
        d_groups,
        num_hedges,
        d_coarse_hedges_offsets,
        d_coarse_srcs_count,
        d_coarse_hedges
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(d_coarse_hedges_buffer));
    CUDA_CHECK(cudaFree(c_hedges_storage));

    return std::make_tuple(new_hedges_size, new_max_hedge_size, d_coarse_hedges, d_coarse_hedges_offsets, d_coarse_srcs_count);
}

std::tuple<dim_t, uint32_t*, dim_t*, uint32_t*> coarsenTouching(
    const runconfig cfg,
    const uint32_t *d_coarse_hedges,
    const dim_t *d_coarse_hedges_offsets,
    const uint32_t *d_touching,
    const dim_t *d_touching_offsets,
    const uint32_t *d_inbound_count,
    const uint32_t *d_ungroups,
    const dim_t *d_ungroups_offsets,
    const uint32_t new_num_nodes,
    const uint32_t num_hedges
) {
    uint32_t *d_coarse_touching = nullptr;
    uint32_t *d_coarse_touching_buffer = nullptr;
    dim_t *d_coarse_touching_offsets = nullptr;
    uint32_t *d_coarse_inbound_count = nullptr;

    CUDA_CHECK(cudaMalloc(&d_coarse_touching_offsets, (1 + new_num_nodes) * sizeof(dim_t))); // NOTE: the number nodes decreases!
    CUDA_CHECK(cudaMemset(d_coarse_touching_offsets, 0x00, (1 + new_num_nodes) * sizeof(dim_t))); // remember to leave the first offset at 0
    CUDA_CHECK(cudaMalloc(&d_coarse_inbound_count, new_num_nodes * sizeof(uint32_t)));

    {
        // launch configuration - coarsening kernel (touching - count)
        int threads_per_block = 128;
        int num_threads_needed = num_hedges; // 1 thread per hedge
        int blocks = (num_threads_needed + threads_per_block - 1) / threads_per_block;
        // launch - coarsening kernel (touching - count)
        std::cout << "Running coarsening kernel (touching - count) (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
        apply_coarsening_touching_count<<<blocks, threads_per_block>>>(
            d_coarse_hedges,
            d_coarse_hedges_offsets,
            num_hedges,
            d_coarse_touching_offsets
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    thrust::device_ptr<dim_t> t_coarse_touching_offsets(d_coarse_touching_offsets);
    thrust::inclusive_scan(t_coarse_touching_offsets, t_coarse_touching_offsets + (new_num_nodes + 1), t_coarse_touching_offsets); // in-place exclusive scan (the last element is set to zero and thus collects the full reduce)
    dim_t new_touching_size = 0; // last value in the inclusive scan = full reduce = total number of touching hedges among all sets
    CUDA_CHECK(cudaMemcpy(&new_touching_size, d_coarse_touching_offsets + new_num_nodes, sizeof(dim_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMalloc(&d_coarse_touching, new_touching_size * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_coarse_touching_buffer, new_touching_size * sizeof(uint32_t)));

    {
        // launch configuration - coarsening kernel (touching - scatter - inbound)
        int threads_per_block = 128; // 128/32 -> 4 warps per block
        int warps_per_block = threads_per_block / WARP_SIZE;
        int num_warps_needed = new_num_nodes; // 1 warp per group
        int blocks = (num_warps_needed + warps_per_block - 1) / warps_per_block;
        // compute shared memory per block (bytes)
        size_t bytes_per_warp = MAX_SM_WARP_DEDUPE_BUFFER_SIZE * sizeof(uint32_t);
        size_t shared_bytes = warps_per_block * bytes_per_warp;
        // launch - coarsening kernel (touching - scatter - inbound)
        std::cout << "Running coarsening kernel (touching - scatter - inbound) (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
        apply_coarsening_touching_scatter_inbound<<<blocks, threads_per_block, shared_bytes>>>(
            d_touching,
            d_touching_offsets,
            d_inbound_count,
            d_ungroups,
            d_ungroups_offsets,
            new_num_nodes,
            d_coarse_touching_offsets,
            d_coarse_touching,
            d_coarse_inbound_count
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // sort each inbound touching set
        cub::DoubleBuffer<uint32_t> c_coarse_touching_double_buffer(d_coarse_touching, d_coarse_touching_buffer);
        void* c_touching_storage = nullptr;
        size_t c_touching_storage_bytes = 0;
        cub::DeviceSegmentedRadixSort::SortKeys(c_touching_storage, c_touching_storage_bytes, c_coarse_touching_double_buffer, new_touching_size, new_num_nodes, d_coarse_touching_offsets, d_coarse_touching_offsets + 1, /*begin_bit=*/0, /*end_bit=*/sizeof(uint32_t) * 8, /*stream*/0);
        std::cout << "CUB segmented sort requiring " << std::fixed << std::setprecision(3) << (float)(new_touching_size * sizeof(uint32_t)) / (1 << 30) << " GB of pong-buffer and " << std::fixed << std::setprecision(3) << ((float)c_touching_storage_bytes) / (1 << 20) << " MB of temporary storage ...\n";
        cudaMalloc(&c_touching_storage, c_touching_storage_bytes);
        cub::DeviceSegmentedRadixSort::SortKeys(c_touching_storage, c_touching_storage_bytes, c_coarse_touching_double_buffer, new_touching_size, new_num_nodes, d_coarse_touching_offsets, d_coarse_touching_offsets + 1, /*begin_bit=*/0, /*end_bit=*/sizeof(uint32_t) * 8, /*stream*/0);
        if (c_coarse_touching_double_buffer.Current() != d_coarse_touching) {
            uint32_t* tmp = d_coarse_touching_buffer;
            d_coarse_touching_buffer = d_coarse_touching;
            d_coarse_touching = tmp;
        }
        CUDA_CHECK(cudaFree(d_coarse_touching_buffer));
        CUDA_CHECK(cudaFree(c_touching_storage));
        
        // launch configuration - coarsening kernel (touching - scatter - outbound) - same as coarsening kernel (touching - scatter - inbound)
        // launch - coarsening kernel (touching - scatter - outbound)
        std::cout << "Running coarsening kernel (touching - scatter - outbound) (blocks=" << blocks << ", thr-per-block=" << threads_per_block << ") ...\n";
        apply_coarsening_touching_scatter_outbound<<<blocks, threads_per_block, shared_bytes>>>(
            d_touching,
            d_touching_offsets,
            d_inbound_count,
            d_ungroups,
            d_ungroups_offsets,
            new_num_nodes,
            d_coarse_touching_offsets,
            d_coarse_inbound_count,
            d_coarse_touching
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    return std::make_tuple(new_touching_size, d_coarse_touching, d_coarse_touching_offsets, d_coarse_inbound_count);
}
