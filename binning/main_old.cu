// bin_compare.cu
//
// Compare 3 CUDA/thrust/CUB-based ways to "bin"/group an array by a bin id array.
//
// Notes (important):
// - Target is ~1 GiB of *values* (int32). bins are uint8 (0..K-1).
// - Method (1) as specified allocates K full-size prefix arrays (N * 4 bytes each).
//   For N ~= 268M and K=8, that's ~8 GiB just for the prefix arrays (plus other buffers),
//   which exceeds many GPUs. This program will *auto-downscale N at runtime* if needed,
//   while keeping the target constants in the source.
// - "Scan by key" (thrust::exclusive_scan_by_key / cub::DeviceScan::ExclusiveScanByKey)
//   is a *segmented scan over consecutive equal keys*. With random bins, you must first
//   group (sort) by key for it to be meaningful. Method (2) therefore sorts a *copy*
//   of keys+indices, then does scan_by_key, then scatters into the final out-of-place
//   array using per-bin base offsets derived from the segment totals.
//
// Build (example):
//   nvcc -O3 -std=c++17 bin_compare.cu -o bin_compare
//
// Run:
//   ./bin_compare
//
// (If you want to pin N, remove the downscale logic and ensure your GPU has enough memory.)

// Compile instructions:
// $ nvcc -arch=native -allow-unsupported-compiler --extended-lambda main.cu -o main.exe -run

#include </home/mronzani/cuda/include/cuda_runtime.h>

#include </home/mronzani/cuda/include/cub/cub.cuh>

#include </home/mronzani/cuda/include/thrust/device_vector.h>
#include </home/mronzani/cuda/include/thrust/host_vector.h>

#include </home/mronzani/cuda/include/thrust/copy.h>
#include </home/mronzani/cuda/include/thrust/count.h>
#include </home/mronzani/cuda/include/thrust/execution_policy.h>
#include </home/mronzani/cuda/include/thrust/fill.h>
#include </home/mronzani/cuda/include/thrust/functional.h>
#include </home/mronzani/cuda/include/thrust/gather.h>
#include </home/mronzani/cuda/include/thrust/iterator/counting_iterator.h>
#include </home/mronzani/cuda/include/thrust/iterator/zip_iterator.h>
#include </home/mronzani/cuda/include/thrust/random.h>
#include </home/mronzani/cuda/include/thrust/reduce.h>
#include </home/mronzani/cuda/include/thrust/scan.h>
#include </home/mronzani/cuda/include/thrust/sort.h>
#include </home/mronzani/cuda/include/thrust/transform.h>

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <vector>

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

static float time_ms(std::function<void()> fn) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    fn();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms;
}

template <typename T>
struct FillRandIntFunctor {
    uint64_t seed;
    int lo, hi; // inclusive
    __host__ __device__ T operator()(uint64_t idx) const {
        thrust::default_random_engine rng(static_cast<unsigned int>(seed));
        rng.discard(static_cast<unsigned int>(idx));
        thrust::uniform_int_distribution<int> dist(lo, hi);
        return static_cast<T>(dist(rng));
    }
};

struct SetFlagIfEq {
    uint8_t target;
    const uint8_t* bins;
    __host__ __device__ uint32_t operator()(uint64_t idx) const {
        return (bins[idx] == target) ? 1u : 0u;
    }
};

struct ScatterMethod1Functor {
    const int* in;
    const uint8_t* bins;
    int* out;

    const int* base;                 // [K]
    const uint32_t* const* offsets;  // [K] pointers to [N]
    uint64_t N;

    __host__ __device__ void operator()(uint64_t idx) const {
        uint8_t b = bins[idx];
        uint32_t off = offsets[b][idx];
        int pos = base[b] + static_cast<int>(off);
        out[pos] = in[idx];
    }
};

struct ScatterByRankFunctor {
    // used by method 2: after sorting by bins we have:
    //  - sorted_bins[pos] in [0..K-1]
    //  - sorted_idx[pos] is original index
    //  - rank_in_bin[pos] is exclusive rank within that bin segment
    // base[bin] is starting offset of the bin in final output
    const int* in;
    const uint8_t* sorted_bins;
    const uint32_t* sorted_idx;
    const uint32_t* rank_in_bin;
    const int* base; // [K]
    int* out;

    __host__ __device__ void operator()(uint64_t pos) const {
        uint8_t b = sorted_bins[pos];
        uint32_t r = rank_in_bin[pos];
        int dst = base[b] + static_cast<int>(r);
        uint32_t src = sorted_idx[pos];
        out[dst] = in[src];
    }
};

static void compute_counts_host(const thrust::device_vector<uint8_t>& bins, int K, std::vector<int>& counts_out) {
    counts_out.assign(K, 0);
    for (int k = 0; k < K; ++k)
        counts_out[k] = static_cast<int>(thrust::count(bins.begin(), bins.end(), static_cast<uint8_t>(k)));
}

static bool validate_grouped_bins(const thrust::device_vector<uint8_t>& out_bins, const std::vector<int>& expected_counts, int K) {
    // 1) bins must be nondecreasing
    bool sorted = thrust::is_sorted(out_bins.begin(), out_bins.end());
    if (!sorted) return false;

    // 2) per-bin counts must match
    for (int k = 0; k < K; ++k) {
        int c = static_cast<int>(thrust::count(out_bins.begin(), out_bins.end(), static_cast<uint8_t>(k)));
        if (c != expected_counts[k]) return false;
    }
    return true;
}

int main() {
    constexpr uint64_t TARGET_VALUE_BYTES = (1ull << 30); // 1 GiB of *values* (int32)
    constexpr int K = 8;

    int device = 0;
    CUDA_CHECK(cudaSetDevice(device));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    std::cout << "GPU: " << prop.name << "\n";

    uint64_t N_target = TARGET_VALUE_BYTES / sizeof(int);
    if (N_target == 0) {
        std::cerr << "TARGET_VALUE_BYTES too small.\n";
        return 1;
    }

    // Rough runtime memory model to avoid OOM (very conservative).
    // We will allocate:
    //   in:      N * 4
    //   bins:    N * 1
    // plus method-specific scratch (worst case is method 1).
    // Method 1 worst case:
    //   K offsets arrays: N * 4 * K
    //   flags temp: N * 4
    //   out + out_bins: N*4 + N*1
    //
    // Total ~ N*(4 + 1 + 4K + 4 + 4 + 1) = N*(4K + 14) bytes.
    // For K=8 => N*46 bytes. With N=268M, that's ~12.3 GiB, too big on many GPUs.
    size_t freeB = 0, totalB = 0;
    CUDA_CHECK(cudaMemGetInfo(&freeB, &totalB));
    std::cout << "Device memory: free " << (freeB / (1024.0 * 1024.0 * 1024.0)) << " GiB / total " << (totalB / (1024.0 * 1024.0 * 1024.0)) << " GiB\n";

    auto bytes_needed_method1 = [&](uint64_t N) -> uint64_t {
        return N * static_cast<uint64_t>(4ull * K + 14ull);
    };

    uint64_t N = N_target;
    // Keep a safety margin (leave 10% free).
    uint64_t budget = static_cast<uint64_t>(freeB * 0.90);
    if (bytes_needed_method1(N) > budget) {
        uint64_t N_new = budget / (4ull * K + 14ull);
        // round down to multiple of 256 for nicer kernels
        N_new = (N_new / 256) * 256;
        if (N_new < 1024) {
            std::cerr << "Not enough GPU memory even for a tiny run.\n";
            return 1;
        }
        std::cout << "WARNING: Auto-downscaling N from " << N << " to " << N_new << " due to GPU memory limits (method 1 footprint).\n";
        N = N_new;
    }

    std::cout << "N = " << N << " elements (values bytes ~= " << (N * sizeof(int) / (1024.0 * 1024.0 * 1024.0))
                        << " GiB)\n";
    std::cout << "K = " << K << "\n\n";

    // -------------------------
    // Setup input arrays on device
    // -------------------------
    thrust::device_vector<int> d_in(N);
    thrust::device_vector<uint8_t> d_bins(N);

    {
        auto it = thrust::make_counting_iterator<uint64_t>(0);
        uint64_t seed = 1234567ull;

        // Fill values with random int32
        thrust::transform(it, it + N, d_in.begin(), FillRandIntFunctor<int>{seed, -1000000, 1000000});

        // Fill bins with random uint8 in [0..K-1]
        thrust::transform(it, it + N, d_bins.begin(), FillRandIntFunctor<uint8_t>{seed ^ 0x9e3779b97f4a7c15ull, 0, K - 1});
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Precompute expected per-bin counts from the original bins (for validation).
    std::vector<int> expected_counts;
    compute_counts_host(d_bins, K, expected_counts);

    // Print counts summary
    {
        std::cout << "Original bin counts:\n";
        for (int k = 0; k < K; ++k)
            std::cout << "  bin " << k << ": " << expected_counts[k] << "\n";
        std::cout << "\n";
    }

    // Outputs for each method
    thrust::device_vector<int> out1, out2, out3;
    thrust::device_vector<uint8_t> out1_bins, out2_bins, out3_bins;

    float ms1 = 0, ms2 = 0, ms3 = 0;

    // ============================================================
    // Method 1:
    // K scans over bins (one scan per bin), allocate K device arrays.
    // ============================================================
    ms1 = time_ms([&]() {
        out1.assign(N, 0);
        out1_bins.assign(N, 0);

        // Per-bin offsets (K arrays of length N)
        std::vector<thrust::device_vector<uint32_t>> d_offsets_vec;
        d_offsets_vec.reserve(K);
        for (int k = 0; k < K; ++k) d_offsets_vec.emplace_back(N);

        // Temporary flags
        thrust::device_vector<uint32_t> d_flags(N);

        // Compute per-bin offsets and counts
        std::vector<int> counts(K, 0);

        for (int k = 0; k < K; ++k) {
            // flags[idx] = (bins[idx] == k) ? 1 : 0
            thrust::transform(
                thrust::make_counting_iterator<uint64_t>(0),
                thrust::make_counting_iterator<uint64_t>(0) + N,
                d_flags.begin(),
                SetFlagIfEq{static_cast<uint8_t>(k), thrust::raw_pointer_cast(d_bins.data())}
            );

            // exclusive scan flags -> offsets[k]
            thrust::exclusive_scan(d_flags.begin(), d_flags.end(), d_offsets_vec[k].begin(), 0u);

            // bin count
            uint32_t last_flag, last_offset;
            CUDA_CHECK(cudaMemcpy(&last_flag, thrust::raw_pointer_cast(d_flags.data()) + (N - 1), sizeof(uint32_t), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&last_offset, thrust::raw_pointer_cast(d_offsets_vec[k].data()) + (N - 1), sizeof(uint32_t), cudaMemcpyDeviceToHost));
            counts[k] = last_offset + last_flag;
        }

        // Compute base offsets (host), then copy to device
        std::vector<int> base_h(K, 0);
        int running = 0;
        for (int k = 0; k < K; ++k) {
            base_h[k] = running;
            running += counts[k];
        }
        thrust::device_vector<int> d_base = base_h;

        // Build device array of pointers to per-bin offset arrays
        std::vector<const uint32_t*> offset_ptrs_h(K, nullptr);
        for (int k = 0; k < K; ++k) offset_ptrs_h[k] = thrust::raw_pointer_cast(d_offsets_vec[k].data());

        thrust::device_vector<const uint32_t*> d_offset_ptrs(K);
        thrust::copy(offset_ptrs_h.begin(), offset_ptrs_h.end(), d_offset_ptrs.begin());

        // Scatter (parallel for_each over indices)
        ScatterMethod1Functor f{
            thrust::raw_pointer_cast(d_in.data()),
            thrust::raw_pointer_cast(d_bins.data()),
            thrust::raw_pointer_cast(out1.data()),
            thrust::raw_pointer_cast(d_base.data()),
            thrust::raw_pointer_cast(d_offset_ptrs.data()),
            N
        };

        thrust::for_each(
            thrust::make_counting_iterator<uint64_t>(0),
            thrust::make_counting_iterator<uint64_t>(0) + N,
            f
        );

        // Also scatter bins (for validation) using the exact same positions:
        // We reuse the same functor idea via a second for_each.
        auto* bins_ptr = thrust::raw_pointer_cast(d_bins.data());
        auto* out_bins_ptr = thrust::raw_pointer_cast(out1_bins.data());
        auto* base_ptr = thrust::raw_pointer_cast(d_base.data());
        auto* offs_ptr = thrust::raw_pointer_cast(d_offset_ptrs.data());

        thrust::for_each(
            thrust::make_counting_iterator<uint64_t>(0),
            thrust::make_counting_iterator<uint64_t>(0) + N,
            [=] __host__ __device__ (uint64_t idx) {
                uint8_t b = bins_ptr[idx];
                uint32_t off = offs_ptr[b][idx];
                int pos = base_ptr[b] + static_cast<int>(off);
                out_bins_ptr[pos] = b;
            }
        );

        CUDA_CHECK(cudaDeviceSynchronize());
    });

    // ============================================================
    // Method 2:
    // "Scan by key" (requires grouping keys first, since scan_by_key is segmented by adjacent keys).
    //
    // Steps:
    //  - Make copy of (bins, idx)
    //  - Sort by bins (keys), permuting idx
    //  - exclusive_scan_by_key over an array of 1s to compute rank within each bin segment
    //  - derive per-bin sizes (segment totals) and base offsets
    //  - scatter input into final output using (base[bin] + rank)
    // ============================================================
    ms2 = time_ms([&]() {
        out2.assign(N, 0);
        out2_bins.assign(N, 0);

        thrust::device_vector<uint8_t> bins_sorted = d_bins;
        thrust::device_vector<uint32_t> idx_sorted(N);
        thrust::sequence(idx_sorted.begin(), idx_sorted.end(), 0u);

        // sort by key (bins), permuting indices
        thrust::sort_by_key(bins_sorted.begin(), bins_sorted.end(), idx_sorted.begin());

        // ones array (uint32)
        thrust::device_vector<uint32_t> ones(N, 1u);
        thrust::device_vector<uint32_t> rank_in_bin(N, 0u);

        // segmented exclusive scan by key
        thrust::exclusive_scan_by_key(
            bins_sorted.begin(),
            bins_sorted.end(),
            ones.begin(),
            rank_in_bin.begin(),
            0u
        );

        // Compute per-bin counts from the sorted keys by counting (K small)
        std::vector<int> counts(K, 0);
        for (int k = 0; k < K; ++k)
            counts[k] = static_cast<int>(thrust::count(bins_sorted.begin(), bins_sorted.end(), static_cast<uint8_t>(k)));

        // base offsets
        std::vector<int> base_h(K, 0);
        int running = 0;
        for (int k = 0; k < K; ++k) {
            base_h[k] = running;
            running += counts[k];
        }
        thrust::device_vector<int> d_base = base_h;

        // Scatter using (base[bin] + rank)
        ScatterByRankFunctor f{
            thrust::raw_pointer_cast(d_in.data()),
            thrust::raw_pointer_cast(bins_sorted.data()),
            thrust::raw_pointer_cast(idx_sorted.data()),
            thrust::raw_pointer_cast(rank_in_bin.data()),
            thrust::raw_pointer_cast(d_base.data()),
            thrust::raw_pointer_cast(out2.data())
        };

        thrust::for_each(
            thrust::make_counting_iterator<uint64_t>(0),
            thrust::make_counting_iterator<uint64_t>(0) + N,
            f
        );

        // Produce out2_bins for validation (same mapping)
        auto* in_ptr = thrust::raw_pointer_cast(d_in.data()); (void)in_ptr;
        auto* bs_ptr = thrust::raw_pointer_cast(bins_sorted.data());
        auto* is_ptr = thrust::raw_pointer_cast(idx_sorted.data()); (void)is_ptr;
        auto* r_ptr = thrust::raw_pointer_cast(rank_in_bin.data());
        auto* b_ptr = thrust::raw_pointer_cast(d_base.data());
        auto* ob_ptr = thrust::raw_pointer_cast(out2_bins.data());

        thrust::for_each(
            thrust::make_counting_iterator<uint64_t>(0),
            thrust::make_counting_iterator<uint64_t>(0) + N,
            [=] __host__ __device__ (uint64_t pos) {
                uint8_t b = bs_ptr[pos];
                uint32_t r = r_ptr[pos];
                int dst = b_ptr[b] + static_cast<int>(r);
                ob_ptr[dst] = b;
            }
        );

        CUDA_CHECK(cudaDeviceSynchronize());
    });

    // ============================================================
    // Method 3:
    // Sort by key using bins as keys and indices as values, then gather.
    // ============================================================
    ms3 = time_ms([&]() {
        out3.assign(N, 0);
        out3_bins.assign(N, 0);

        thrust::device_vector<uint8_t> bins_sorted = d_bins;
        thrust::device_vector<uint32_t> idx_sorted(N);
        thrust::sequence(idx_sorted.begin(), idx_sorted.end(), 0u);

        thrust::sort_by_key(bins_sorted.begin(), bins_sorted.end(), idx_sorted.begin());

        // Gather values into final order
        thrust::gather(idx_sorted.begin(), idx_sorted.end(), d_in.begin(), out3.begin());

        // out3_bins is just the sorted keys
        out3_bins = bins_sorted;

        CUDA_CHECK(cudaDeviceSynchronize());
    });

    // -------------------------
    // Validation
    // -------------------------
    bool ok1 = validate_grouped_bins(out1_bins, expected_counts, K);
    bool ok2 = validate_grouped_bins(out2_bins, expected_counts, K);
    bool ok3 = validate_grouped_bins(out3_bins, expected_counts, K);

    std::cout << "Timing (ms):\n";
    std::cout << "  Method 1 (K scans + scatter):            " << ms1 << " ms\n";
    std::cout << "  Method 2 (sort + scan_by_key + scatter): " << ms2 << " ms\n";
    std::cout << "  Method 3 (sort_by_key + gather):         " << ms3 << " ms\n\n";

    std::cout << "Validation:\n";
    std::cout << "  Method 1: " << (ok1 ? "OK" : "FAIL") << "\n";
    std::cout << "  Method 2: " << (ok2 ? "OK" : "FAIL") << "\n";
    std::cout << "  Method 3: " << (ok3 ? "OK" : "FAIL") << "\n";

    if (!(ok1 && ok2 && ok3)) {
        std::cerr << "At least one method failed validation.\n";
        return 2;
    }

    return 0;
}
