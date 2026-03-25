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
#include <vector>
#include <cstdint>
#include <cstdlib>
#include <numeric>
#include <iomanip>
#include <iostream>

#define HOST_CHECK false

#define CUDA_CHECK(ans) do { gpuAssert((ans), __FILE__, __LINE__); } while (0)
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        std::fprintf(stderr, "CUDA error: %s %s:%d\n", cudaGetErrorString(code), file, line);
        if (abort) std::exit((int)code);
    }
}

static float time_ms(const std::function<void()>& fn) {
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
        thrust::default_random_engine rng((unsigned int)seed);
        rng.discard((unsigned int)idx);
        thrust::uniform_int_distribution<int> dist(lo, hi);
        return (T)dist(rng);
    }
};

int main(int argc, char** argv) {
    // defaults
    uint64_t SIZE = (1ull << 30); // in bytes, default is 1 GiB of int32s
    int K = 8;

    if (argc > 1) {
        if (std::strcmp(argv[1], "-h") == 0 || std::strcmp(argv[1], "--help") == 0) {
            std::cout << "Usage: " << argv[0] << " [SIZE] [K]\n"
                    << "  SIZE : total GiBs of input values (default 1GiB)\n"
                    << "  K    : number of bins (default 8)\n";
            return 0;
        }
        SIZE = (uint64_t)(std::strtof(argv[1], nullptr) * (float)(1ull << 30));
    }
    if (argc > 2) {
        K = std::atoi(argv[2]);
    }

    if (SIZE == 0 || K <= 0) {
        std::cerr << "Invalid arguments. Use --help for usage.\n";
        return 1;
    }

    uint64_t N = SIZE / sizeof(int);
    if (N < 1024) {
        std::cerr << "SIZE too small (min 1024).\n";
        return 1;
    }

    if (K > 256) {
        std::cerr << "K too large (max 256).\n";
        return 1;
    }

    std::cout << "Settings:\n"
            << "  SIZE = " << SIZE << " B\n"
            << "  K    = " << K << " bins\n"
            << "  N    = " << N << " values\n\n";

    int device = 0;
    CUDA_CHECK(cudaSetDevice(device));

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    std::cout << "GPU: " << prop.name << "\n\n";

    // setup input arrays
    thrust::device_vector<int> d_in(N);
    thrust::device_vector<uint8_t> d_bins(N);

    {
        auto it = thrust::make_counting_iterator<uint64_t>(0);
        uint64_t seed = 1234567ull;
        thrust::transform(it, it + N, d_in.begin(), FillRandIntFunctor<int>{seed, -1000000, 1000000});
        thrust::transform(it, it + N, d_bins.begin(), FillRandIntFunctor<uint8_t>{seed ^ 0x9e3779b97f4a7c15ull, 0, K - 1});
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // expected counts per bin
    std::vector<int> expected_counts(K, 0);
    for (int k = 0; k < K; ++k)
        expected_counts[k] = (int)thrust::count(d_bins.begin(), d_bins.end(), (uint8_t)k);

    std::cout << "Original bin counts:\n";
    for (int k = 0; k < K; ++k)
        std::cout << "  bin " << k << ": " << expected_counts[k] << "\n";
    std::cout << "\n";

    // outputs
    thrust::device_vector<int> out1(N), out2(N);
    thrust::device_vector<uint8_t> out1_bins(N), out2_bins(N);

    // pre-allocate method 1's buffers before timing
    std::vector<thrust::device_vector<uint32_t>> offsets;
    offsets.reserve(K);
    for (int k = 0; k < K; ++k) offsets.emplace_back(N);

    thrust::device_vector<int> d_base(K);
    thrust::device_vector<const uint32_t*> d_offset_ptrs(K);

    std::vector<int> base_h(K, 0);
    std::vector<int> counts_h(K, 0);
    std::vector<const uint32_t*> offset_ptrs_h(K, nullptr);

    /*
    * Method 1:
    * - K scans over bins, one scan per bin, in its own array
    * - scatter based on each element's bin's array
    */
    std::cout << "Running method 1 ...\n";
    float ms1 = time_ms([&]() {
        // bin of the last element (to correct its exclusive scan's final count)
        uint8_t last_bin = 0;
        CUDA_CHECK(cudaMemcpy(&last_bin, thrust::raw_pointer_cast(d_bins.data()) + (N - 1), sizeof(uint8_t), cudaMemcpyDeviceToHost));
        
        // one SCAN per bin
        // => build per-bin offsets
        for (int k = 0; k < K; ++k) {
            // flags: 1 for elements with bin "k", 0 otherwise
            auto flags_it = thrust::make_transform_iterator(
                d_bins.begin(),
                [=] __host__ __device__ (uint8_t b) -> uint32_t { return (b == (uint8_t)k) ? 1u : 0u; }
            );

            /*
            * Potential upgrade:
            * - change from one scan per bin, to one scan handling multiple bins
            * - custom scan kernel, carrying around an array of prefixes, instead of a scalar value, and switching the prefix to update on each element depending on its bin
            * => trade-off: while this would reduce the number of read passes over bins, it multiplies equally the cost of moving around prefixes
            */

            // exclusive scan flags into offsets
            thrust::exclusive_scan(flags_it, flags_it + N, offsets[k].begin(), 0u);

            uint32_t last_offset = 0;
            CUDA_CHECK(cudaMemcpy(&last_offset, thrust::raw_pointer_cast(offsets[k].data()) + (N - 1), sizeof(uint32_t), cudaMemcpyDeviceToHost));
            // the last exclusive scan value gives you the count, unless you are the bin owning the last element, then +1
            counts_h[k] = (int)(last_offset + ((last_bin == (uint8_t)k) ? 1u : 0u));
        }

        // pre-compute base offsets on the host
        // => running a kernel for < 1k elements is pointless
        int running = 0;
        for (int k = 0; k < K; ++k) {
            base_h[k] = running;
            running += counts_h[k];
        }
        thrust::copy(base_h.begin(), base_h.end(), d_base.begin());

        // build a single device array of pointers to offsets arrays
        for (int k = 0; k < K; ++k)
            offset_ptrs_h[k] = thrust::raw_pointer_cast(offsets[k].data());
        thrust::copy(offset_ptrs_h.begin(), offset_ptrs_h.end(), d_offset_ptrs.begin());

        // scatter values and bins over to their offsets
        const int* in_ptr = thrust::raw_pointer_cast(d_in.data());
        const uint8_t* bins_ptr = thrust::raw_pointer_cast(d_bins.data());
        int* out_ptr = thrust::raw_pointer_cast(out1.data());
        uint8_t* out_bins_ptr = thrust::raw_pointer_cast(out1_bins.data());
        const int* base_ptr = thrust::raw_pointer_cast(d_base.data());
        const uint32_t* const* offs_ptr = thrust::raw_pointer_cast(d_offset_ptrs.data());
        thrust::for_each(
            thrust::make_counting_iterator<uint64_t>(0),
            thrust::make_counting_iterator<uint64_t>(0) + N,
            [=] __host__ __device__ (uint64_t i) {
                uint8_t b = bins_ptr[i];
                uint32_t r = offs_ptr[b][i];
                int dst = base_ptr[b] + (int)r;
                out_ptr[dst] = in_ptr[i];
                out_bins_ptr[dst] = b;
            }
        );

        CUDA_CHECK(cudaDeviceSynchronize());
    });

    /*
    * Method 2:
    * - sort by key: sort bins while carrying indices along
    * - gather based on rearranged indices
    * 
    * Potential upgrade: direclty sort by key using (bin, input) pairs, no gather, no indices
    * Issues:
    * - not stable, each bin gets re-ordered unless its content is masked
    * - more expensive if inputs are larger than 32bit values
    */
    std::cout << "Running method 2 ...\n";
    float ms2 = time_ms([&]() {
        thrust::device_vector<uint8_t> bins_sorted = d_bins;
        thrust::device_vector<uint32_t> idx_sorted(N);
        // array of indices
        thrust::sequence(idx_sorted.begin(), idx_sorted.end(), 0u);

        thrust::sort_by_key(bins_sorted.begin(), bins_sorted.end(), idx_sorted.begin());
        thrust::gather(idx_sorted.begin(), idx_sorted.end(), d_in.begin(), out2.begin());
        out2_bins = std::move(bins_sorted);

        CUDA_CHECK(cudaDeviceSynchronize());
    });

    // validation
    // => here valid means bins are grouped as 0s then 1s ... then K-1s, that is equivalent to a non-decreasing order of bins
    std::cout << "Running validation ...\n";
    bool ok1 = thrust::is_sorted(out1_bins.begin(), out1_bins.end());;
    bool ok2 = thrust::is_sorted(out2_bins.begin(), out2_bins.end());;
    // => secondly, require that bins are correctly preserved (same number of elements per bin)
    for (int k = 0; k < K; ++k) {
        int c1 = (int)thrust::count(out1_bins.begin(), out1_bins.end(), (uint8_t)k);
        int c2 = (int)thrust::count(out2_bins.begin(), out2_bins.end(), (uint8_t)k);
        if (c1 != expected_counts[k]) ok1 = false;
        if (c2 != expected_counts[k]) ok2 = false;
    }
    #if HOST_CHECK
    // => check against binning on the host
    std::cout << "Running HOST SIDE validation ...\n";
    {
        std::vector<int> h_in(N), h_out1(N), h_out2(N), h_ref(N);
        std::vector<uint8_t> h_bins(N), h_out1_bins(N), h_out2_bins(N), h_ref_bins(N);

        thrust::copy(d_in.begin(), d_in.end(), h_in.begin());
        thrust::copy(d_bins.begin(), d_bins.end(), h_bins.begin());
        thrust::copy(out1.begin(), out1.end(), h_out1.begin());
        thrust::copy(out1_bins.begin(), out1_bins.end(), h_out1_bins.begin());
        thrust::copy(out2.begin(), out2.end(), h_out2.begin());
        thrust::copy(out2_bins.begin(), out2_bins.end(), h_out2_bins.begin());

        // per-bin checksum
        for (int k = 0; k < K; ++k) {
            long long sum_ref = 0, sum1 = 0, sum2 = 0;

            for (size_t i = 0; i < N; ++i)
                if (h_bins[i] == k) sum_ref += h_in[i];

            for (size_t i = 0; i < N; ++i) {
                if (h_out1_bins[i] == k) sum1 += h_out1[i];
                if (h_out2_bins[i] == k) sum2 += h_out2[i];
            }

            if (sum1 != sum_ref) ok1 = false;
            if (sum2 != sum_ref) ok2 = false;
        }
    }
    #endif

    std::cout << "\nTiming (ms):\n";
    std::cout << "  Method 1 (K scans + scatter):    " << std::fixed << std::setprecision(6) << ms1 << " ms\n";
    std::cout << "  Method 2 (sort_by_key + gather): " << std::fixed << std::setprecision(6) << ms2 << " ms\n\n";

    std::cout << "Validation:\n";
    std::cout << "  Method 1: " << (ok1 ? "OK" : "FAIL") << "\n";
    std::cout << "  Method 2: " << (ok2 ? "OK" : "FAIL") << "\n";

    if (!(ok1 && ok2)) {
        std::cerr << "At least one method failed validation.\n";
        return 2;
    }

    return 0;
}