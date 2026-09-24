#include <tuple>
#include <vector>
#include <string>
#include <chrono>
#include <format>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <filesystem>
#include <stdexcept>
#include <unordered_map>

#include "runconfig.hpp"

#include "init_part.hpp"

#include "utils.hpp"
#include "prims.hpp"
#include "constants.hpp"

std::tuple<buffer<uint32_t>, buffer<uint32_t>> initial_partitioning_kahypar(
    const runconfig &cfg,
    const uint32_t num_nodes,
    const uint32_t num_hedges,
    const uint32_t* hedges,
    const dim_t* hedges_offsets,
    const float* hedge_weights,
    const dim_t* touching_offsets,
    const dim_t hedges_size,
    const uint32_t* nodes_sizes,
    const uint32_t k,
    const float epsilon
) {
    INFO(cfg) std::cout << "Building initial partitioning via Mt-KaHyPar, remaining nodes=" << num_nodes << ", remaining pins=" << hedges_size << "\n";

    // target: keep n*d hyperedges
    const uint32_t target_keep_hedges = num_nodes * (hedges_size / num_hedges);

    buffer<float> hedge_ratio(num_hedges); // hedge_ratio[hedge idx] -> hedge's weight/score ratio based on the degree of its pins (higher => higher degrees)

    LAUNCH(cfg) RUN << "armonic score kernel (threads=" << cfg.threads << ") ...\n";
    armonic_degree_score_kernel(
        hedges,
        hedges_offsets,
        touching_offsets,
        hedge_weights,
        num_hedges,
        hedge_ratio.data()
    );

    // find the total score
    const float total_score = par_reduce<float>(num_hedges, 0.0f, [&](dim_t i) { return hedge_ratio[i]; }, [](float a, float b) { return a + b; });
    // with this, the expected number of hedges kept is ~~ target_keep_hedges
    const float threshold = target_keep_hedges / total_score;

    std::vector<uint8_t> keep(num_hedges); // keep[hedge idx] -> keep hedge if true
    std::vector<float> hedge_scaled_weights(num_hedges); // hedge_scaled_weights[hedge idx] -> new hedge weight scaled by its retention probability

    LAUNCH(cfg) RUN << "prune hedges kernel (threads=" << cfg.threads << ") ...\n";
    prune_hedges_kernel(
        hedge_weights,
        hedge_ratio.data(),
        num_hedges,
        threshold,
        INIT_SEED,
        hedge_scaled_weights.data(),
        keep.data()
    );
    hedge_ratio.release();

    // copy the hgraph, pins get sorted in place below
    std::vector<uint32_t> hedges_copy(hedges, hedges + hedges_size);

    auto time_start = std::chrono::high_resolution_clock::now();

    // write input file for Mt-KaHyPar
    std::ofstream f_coarse_hg("coarse_tmp.hgr");
    INFO(cfg) std::cout << "Writing temporary file 'coarse_tmp.hgr' ...\n";
    constexpr int HEADER_WIDTH = 64;
    f_coarse_hg << std::setw(HEADER_WIDTH) << std::left << " " << "\n";

    struct HedgeView { uint32_t* data; uint32_t size; };

    struct HedgeHash {
        size_t operator()(const HedgeView& h) const noexcept {
            size_t x = 1469598103934665603ull;
            for (uint32_t i = 0; i < h.size; ++i) {
                x ^= h.data[i];
                x *= 1099511628211ull;
            }
            return x;
        }
    };

    struct HedgeEq {
        bool operator()(const HedgeView& a, const HedgeView& b) const noexcept {
            if (a.size != b.size) return false;
            return std::equal(a.data, a.data + a.size, b.data);
        }
    };

    std::unordered_map<HedgeView, uint64_t, HedgeHash, HedgeEq> map;
    map.reserve(num_hedges);

    // deduplicate hedges
    for (uint32_t i = 0; i < num_hedges; i++) {
        if (!keep[i]) continue;
        uint32_t begin = hedges_offsets[i];
        uint32_t end = hedges_offsets[i + 1];
        uint32_t sz = end - begin;
        if (sz <= 1) continue;
        std::sort(hedges_copy.begin() + begin, hedges_copy.begin() + end);
        HedgeView key{ hedges_copy.data() + begin, sz };
        uint64_t w = static_cast<uint32_t>(hedge_scaled_weights[i] * FIXED_POINT_SCALE);
        map[key] += w;
    }

    // write hedges to file
    uint32_t true_num_hedges = 0u;
    for (auto& [key, weight] : map) {
        f_coarse_hg << static_cast<uint32_t>(weight) << " ";
        for (uint32_t j = 0; j < key.size; ++j) {
            f_coarse_hg << (key.data[j] + 1);
            if (j + 1 < key.size)
                f_coarse_hg << " ";
        }
        f_coarse_hg << "\n";
        ++true_num_hedges;
    }

    // write nodes to file
    for (uint32_t i = 0; i < num_nodes; i++)
        f_coarse_hg << nodes_sizes[i] << "\n";

    // update file header retroactively
    f_coarse_hg.seekp(0);
    std::ostringstream header;
    header << true_num_hedges << " " << num_nodes << " 11";
    f_coarse_hg << std::setw(HEADER_WIDTH) << std::left << header.str() << "\n";

    f_coarse_hg.close();

    INFO(cfg) std::cout << "Preserved unique non-degenerate hedges count: " << true_num_hedges << "\n";

    // invoke Mt-KaHyPar
    std::ostringstream command;
    command
        << "mtkahypar -h coarse_tmp.hgr -k " << k
        << " -e " << std::format("{}", epsilon)
        << " -t " << MAX_OMP_THREADS
        << " -o km1 -v 0 -m direct --write-partition-file 1 --partition-output-folder . --preset-type default --seed " << INIT_SEED;
    INFO(cfg) std::cout << "Running Mt-KaHyPar: " << command.str().c_str() << "\n";
    int command_result = std::system(command.str().c_str());
    std::ostringstream out_filename;
    out_filename << "coarse_tmp.hgr.part" << k << ".epsilon" << std::format("{}", epsilon) << ".seed" << INIT_SEED << ".KaHyPar";
    if (command_result == 0)
        INFO(cfg) std::cout << "Mt-KaHyPar finished successfully ...\n";
    else {
        ERR(cfg) std::cerr << "ERROR, Mt-KaHyPar failed with code: " << command_result << ", inspect 'coarse_tmp.hgr' for possible bugs !!\n";
        abort();
    }
    std::filesystem::remove("coarse_tmp.hgr");

    // parse Mt-KaHyPar output
    buffer<uint32_t> partitions(num_nodes);
    buffer<uint32_t> partitions_sizes(k);
    std::fill(partitions_sizes.data(), partitions_sizes.data() + k, 0u);

    std::ifstream f_parts(out_filename.str().c_str());
    INFO(cfg) std::cout << "Reading temporary file '" << out_filename.str().c_str() << "' ...\n";

    for (uint32_t i = 0; i < num_nodes; i++) {
        if (!(f_parts >> partitions[i])) { // already with 0-based idxs
            ERR(cfg) std::cerr << "ERROR, invalid partitioning returned with " << i << " nodes instead of " << num_nodes << ", inspect '" << out_filename.str().c_str() << "' for possible bugs !!\n";
            throw std::runtime_error("Unexpected partitions file format or too few lines");
        }
        if (partitions[i] >= k) {
            ERR(cfg) std::cerr << "ERROR, partitioning out of range: " << partitions[i] << ", inspect '" << out_filename.str().c_str() << "' for possible bugs !!\n";
            throw std::runtime_error("Partition id out of range");
        }
        partitions_sizes[partitions[i]] += nodes_sizes[i];
    }

    f_parts.close();

    for (uint32_t i = 0; i < k; i++) {
        if (partitions_sizes[i] > max_nodes_per_part) { // already with 0-based idxs
            ERR(cfg) std::cerr << "ERROR, invalid size for partition " << i << " : " << partitions_sizes[i] << " > " << max_nodes_per_part << ", inspect '" << out_filename.str().c_str() << "' for possible bugs !!\n";
            throw std::runtime_error("Invalid partition size");
        }
    }

    std::filesystem::remove_all(out_filename.str().c_str());

    auto time_end = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(time_end - time_start).count();
    INFO(cfg) std::cout << "Host initial partitioning time: " << std::fixed << std::setprecision(3) << total_ms << " ms\n";

    INFO(cfg) std::cout << "Completed initial partitioning via Mt-KaHyPar !\n";
    return std::make_tuple(std::move(partitions), std::move(partitions_sizes));
}
