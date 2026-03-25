#pragma once
#include <set>
#include <cmath>
#include <queue>
#include <vector>
#include <ranges>
#include <utility>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <functional>
#include <unordered_map>

#include <omp.h>

namespace hgraph {

    class HyperEdge;
    class HyperGraph;

    /*
    * HyperEdge:
    * - offset into the HyperGraph's node array
    * - length
    * - sources count
    * - weight
    */
    class HyperEdge {
        public:
        using Node = uint32_t;

        private:
        uint32_t offset_; // offset inside HyperGraph::hedges_flat_
        uint32_t length_; // total nodes = |sources| + |destinations|
        uint32_t src_count_; // how many pins are sources (stored at the start of each segment)
        float weight_;
        const uint32_t* hedges_flat_; // pointer to the owning hypergraph's hedges_flat_ array

        public:
        HyperEdge(uint32_t offset, uint32_t length, uint32_t src_count, float weight, const uint32_t* hedges_flat) : offset_(offset), length_(length), src_count_(src_count), weight_(weight), hedges_flat_(hedges_flat) {}

        uint32_t offset() const { return offset_; }
        uint32_t length() const { return length_; }
        uint32_t src_count() const { return src_count_; }
        float weight() const { return weight_; }
        // aliases for compatibility
        float spikeFrequency() const { return weight_; }
        uint32_t size() const { return length_; }
        uint32_t connections() const { return length_ - 1; }

        void updateWeight(float w) { weight_ = w; }

        // These are "views", the real data lives in HyperGraph.HyperGraph must provide access to hedges_flat_, therefore HyperGraph declares HyperEdge as a "friend".
        std::vector<uint32_t> sources() const {
            return std::vector<uint32_t>(hedges_flat_ + offset_, hedges_flat_ + offset_ + src_count_);
        }
        std::vector<uint32_t> destinations() const {
            return std::vector<uint32_t>(hedges_flat_ + offset_ + src_count_, hedges_flat_ + offset_ + length_);
        }
        std::vector<uint32_t> nodes() const {
            return std::vector<uint32_t>(hedges_flat_ + offset_, hedges_flat_ + offset_ + length_);
        }
    };

    /*
    * Functor for looking up an hyperedge from its index
    */
    struct HedgeLookup {
        const std::vector<HyperEdge>* hedges;

        const HyperEdge* operator()(uint32_t id) const {
            return &(*hedges)[id];
        }
    };
    // type alias for lazy hyperedge lookup iterator
    template<typename Set> using HedgeView = std::ranges::transform_view<std::ranges::ref_view<const Set>,HedgeLookup>;

    /*
    * HyperGraph:
    * - global contiguous array hedges_flat_
    * - hyperedges vector
    * - inbound_/outbound_ adjacency maps
    */
    class HyperGraph {
        public:
        using Node = uint32_t;

        private:
        uint32_t node_count_;

        // contiguous array of nodes in each hyperedge
        std::vector<uint32_t> hedges_flat_;

        // each hyperedge points into hedges_flat_ with the right offset
        std::vector<HyperEdge> hedges_;

        // pointer-based adjacency maps
        std::vector<std::set<uint32_t>> outbound_;  // store indices w.r.t. the hedges array
        std::vector<std::set<uint32_t>> inbound_;

        // contiguous array of the neighbors to each node
        // NOTE: these are only built when needed via a call to "buildNeighborhoods"
        std::vector<uint32_t> neighborhoods_;
        std::vector<uint32_t> neighborhood_offsets_;
        bool neighborhoods_built_ = false;

        public:
        // constructor:
        // - nodes: number of nodes
        // - hedges: vector of pairs of two vectors, srcs and dsts, like: [([src1, src2, ...], [dst1, dst2, ...]), (...), ...]
        // - weights: same length as hedges, one weight per hyperedge.
        HyperGraph(uint32_t nodes, const std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t>>>& hedges, const std::vector<float>& weights) : node_count_(nodes) {
            if (hedges.size() != weights.size())
                throw std::runtime_error("Number of hyperedges != number of weights");

            // compute total nodes needed
            size_t total_pins = 0;
            uint32_t no_src_warning = 0;
            for (auto& he : hedges) {
                if (he.second.size() < 1) throw std::runtime_error("Hyperedge must have at least 1 destination");
                if (he.first.size() < 1) no_src_warning++;
                total_pins += he.first.size() + he.second.size();
            }
            if (no_src_warning) std::cerr << "WARNING: found " << no_src_warning << " hyperedges with zero sources (make sure this is intended - e.g. you loaded an hypergraph in hMETIS format) !!\n";

            hedges_flat_.reserve(total_pins);
            hedges_.reserve(hedges.size());

            // build edges and contiguous array
            uint32_t offset = 0;
            for (size_t i = 0; i < hedges.size(); i++) {
                const auto& he = hedges[i];
                float w = weights[i];

                for (uint32_t v : he.first)
                    hedges_flat_.push_back(v);
                for (uint32_t v : he.second)
                    hedges_flat_.push_back(v);
                    
                hedges_.emplace_back(offset, he.first.size() + he.second.size(), he.first.size(), w, hedges_flat_.data());
                offset += he.first.size() + he.second.size();
            }

            // build inbound/outbound maps
            outbound_.resize(nodes);
            inbound_.resize(nodes);

            for (uint32_t idx = 0; idx < hedges_.size(); idx++) {
                auto& he = hedges_[idx];

                for (uint32_t src : he.sources()) {
                    if (src >= nodes) throw std::runtime_error("Invalid source node");
                    outbound_[src].insert(idx);
                }

                for (uint32_t dst : he.destinations()) {
                    if (dst >= nodes) throw std::runtime_error("Invalid destination node");
                    inbound_[dst].insert(idx);
                }
            }
        }

        uint32_t nodes() const { return node_count_; }
        const std::vector<HyperEdge>& hedges() const { return hedges_; }
        const std::vector<uint32_t>& hedgesFlat() const { return hedges_flat_; }

        // these getters give the set of HyperEdges, lazily paying the de-indexing overhead
        HedgeView<std::set<uint32_t>> outbound(Node n) const {
            if (n >= node_count_) throw std::runtime_error("Invalid node");
            return std::ranges::transform_view(std::ranges::ref_view{ outbound_[n] }, HedgeLookup{ &hedges_ });
        }
        // |
        HedgeView<std::set<uint32_t>> inbound(Node n) const {
            if (n >= node_count_) throw std::runtime_error("Invalid node");
            return std::ranges::transform_view(std::ranges::ref_view{ inbound_[n] }, HedgeLookup{ &hedges_ });
        }

        // these getters give you raw hedge ids w.r.t. the hedges array -> use hg.hedges()[id] to get the actual HyperEdge
        const std::set<uint32_t>& outboundIds(Node n) const {
            if (n >= node_count_) throw std::runtime_error("Invalid node");
            return outbound_[n];
        }
        // |
        const std::set<uint32_t>& inboundIds(Node n) const {
            if (n >= node_count_) throw std::runtime_error("Invalid node");
            return inbound_[n];
        }

        // like outboundIds/inboundIds, these getters give you raw hedge ids w.r.t. the hedges array, also sorting them for each node
        // => std::set always stores elements sorted by its comparator so this is easy!
        const std::ranges::subrange<std::set<uint32_t>::const_iterator> outboundSortedIds(Node n) const {
            if (n >= node_count_) throw std::runtime_error("Invalid node");
            return {outbound_[n].begin(), outbound_[n].end()};
        }
        // |
        const std::ranges::subrange<std::set<uint32_t>::const_iterator> inboundSortedIds(Node n) const {
            if (n >= node_count_) throw std::runtime_error("Invalid node");
            return {inbound_[n].begin(), inbound_[n].end()};
        }

        float totalWeight() const {
            float total = 0.0f;
            for (auto& he : hedges_)
                total += he.weight()*(he.length() - 1);
            return total;
        }
        // aliases for compatibility
        // => lambda - 1 metric (computed assuming this to be a partitioned hgraph)
        float connectivity() const { return totalWeight(); }

        // => cut-net metric (computed assuming this to be a partitioned hgraph)
        float cutnet() const {
            float total = 0.0f;
            for (auto& he : hedges_) {
                if (he.length() > 1)
                    total += he.weight();
            }
            return total;
        }

        // => sum of external degrees
        float soed() const {
            float total = 0.0f;
            for (auto& he : hedges_)
                total += he.weight()*he.length();
            return total;
        }

        // => lambda - 1 metric (computed for a given permutation)
        float connectivityFromPart(const std::vector<uint32_t>& part) const {
            if (part.size() != node_count_) throw std::runtime_error("Partition size mismatch");
            float total = 0.0f;
            for (auto& he : hedges_) {
                std::set<uint32_t> parts;
                for (uint32_t node : he.nodes())
                    parts.insert(part[node]);
                total += he.weight()*(parts.size() - 1);
            }
            return total;
        }

        // => cut-net metric (computed for a given permutation)
        float cutnetFromPart(const std::vector<uint32_t>& part) const {
            if (part.size() != node_count_) throw std::runtime_error("Partition size mismatch");
            float total = 0.0f;
            for (auto& he : hedges_) {
                std::set<uint32_t> parts;
                for (uint32_t node : he.nodes())
                    parts.insert(part[node]);
                if (parts.size() > 1)
                    total += he.weight();
            }
            return total;
        }

        // => sum of external degrees (computed from a given permutation)
        float soedFromPart(const std::vector<uint32_t>& part) const {
            if (part.size() != node_count_) throw std::runtime_error("Partition size mismatch");
            float total = 0.0f;
            for (auto& he : hedges_) {
                std::set<uint32_t> parts;
                for (uint32_t node : he.nodes())
                    parts.insert(part[node]);
                total += he.weight()*(parts.size() - 1)*he.length();
            }
            return total;
        }

        // remove_self_cycles can be:
        // 0 : do not remove
        // 1 : remove the source of the cycle
        // 2 : remove the destination of the cycle
        HyperGraph getPartitionsHypergraph(const std::vector<uint32_t>& part, uint8_t remove_self_cycles, bool squish_hyperedges) const {
            if (part.size() != node_count_) throw std::runtime_error("Partition size mismatch");

            uint32_t new_nodes = 0;
            for (uint32_t p : part) new_nodes = std::max(new_nodes, p + 1);

            for (uint32_t i = 0; i < new_nodes; i++) {
                if (std::find(part.begin(), part.end(), i) == part.end())
                    throw std::runtime_error("Partitions must be 0..N sequential");
            }

            std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t>>> new_hedges;
            std::vector<float> new_weights;

            if (!squish_hyperedges) {
                for (auto& he : hedges_) {
                    std::set<uint32_t> srcp;
                    std::set<uint32_t> dstp;

                    for (uint32_t i = 0; i < he.src_count(); i++)
                        srcp.insert(part[hedges_flat_[he.offset() + i]]);
                    
                    for (uint32_t i = he.src_count(); i < he.length(); i++)
                        dstp.insert(part[hedges_flat_[he.offset() + i]]);
                    
                    if (remove_self_cycles == 1) {
                        for (auto dst : dstp)
                            srcp.erase(dst);
                    } else if (remove_self_cycles == 2) {
                        for (auto src : srcp)
                            dstp.erase(src);
                    }

                    if (srcp.size() > 1 || !dstp.empty()) {
                        std::pair<std::vector<uint32_t>, std::vector<uint32_t>> hv;
                        hv.first.insert(hv.first.end(), srcp.begin(), srcp.end());
                        hv.second.insert(hv.second.end(), dstp.begin(), dstp.end());
                        new_hedges.push_back(hv);
                        new_weights.push_back(he.weight());
                    }
                }
            } else {
                std::unordered_map<uint64_t, float> weight_acc;
                std::unordered_map<uint64_t, std::vector<uint32_t>> src_map;
                std::unordered_map<uint64_t, std::vector<uint32_t>> dst_map;

                for (auto& he : hedges_) {
                    std::set<uint32_t> srcp;
                    std::set<uint32_t> dstp;

                    for (uint32_t i = 0; i < he.src_count(); i++)
                        srcp.insert(part[hedges_flat_[he.offset() + i]]);
                    
                    for (uint32_t i = he.src_count(); i < he.length(); i++)
                        dstp.insert(part[hedges_flat_[he.offset() + i]]);
                    
                    if (remove_self_cycles == 1) {
                        for (auto dst : dstp)
                            srcp.erase(dst);
                    } else if (remove_self_cycles == 2) {
                        for (auto src : srcp)
                            dstp.erase(src);
                    }

                    if (srcp.size() <= 1 && dstp.empty()) continue;

                    uint64_t key = 146527;
                    for (auto s : srcp) key ^= std::hash<uint32_t>()(s) + 0x9e3779b97f4a7c15ULL + (key << 6) + (key >> 2);
                    for (auto d : dstp) key ^= std::hash<uint32_t>()(d) + 0x9e3779b97f4a7c15ULL + (key << 6) + (key >> 2);

                    weight_acc[key] += he.weight();
                    src_map[key] = std::vector<uint32_t>(srcp.begin(), srcp.end());;
                    dst_map[key] = std::vector<uint32_t>(dstp.begin(), dstp.end());
                }

                for (auto& it : weight_acc) {
                    uint64_t k = it.first;
                    float w = it.second;

                    std::pair<std::vector<uint32_t>, std::vector<uint32_t>> hv;
                    auto& srcs = src_map[k];
                    hv.first.insert(hv.first.end(), srcs.begin(), srcs.end());
                    auto& dsts = dst_map[k];
                    hv.second.insert(hv.second.end(), dsts.begin(), dsts.end());
                    new_hedges.push_back(hv);
                    new_weights.push_back(w);
                }
            }

            return HyperGraph(new_nodes, new_hedges, new_weights);
        }

        // return a greedy order of nodes that maximizes high-locality
        // -> frontier expansion starting from nodes with minimal inbound connections
        // -> iteratively add to the frontier the node most strongly connected to those already part of it
        std::vector<int32_t> feedForwardOrder() const {
            std::vector<int32_t> new_id(node_count_, -1);
            std::vector<float> priority(node_count_, 0.0f);
            std::vector<bool> active(node_count_, false);

            uint32_t next_id = 0;

            struct pq_entry {
                float priority;
                uint32_t node;
            };

            struct max_cmp {
                bool operator()(const pq_entry& a, const pq_entry& b) const {
                    if (a.priority != b.priority)
                        return a.priority < b.priority; // max-heap
                    return a.node > b.node; // deterministic tie-break
                }
            };

            std::priority_queue<pq_entry, std::vector<pq_entry>, max_cmp> heap;

            auto activate = [&](uint32_t n, float delta) {
                priority[n] += delta;
                heap.push({priority[n], n});
                active[n] = true;
            };

            while (next_id < node_count_) {
                uint32_t min_inbound = UINT32_MAX;
                std::vector<uint32_t> fallback;

                for (uint32_t n = 0; n < node_count_; ++n) {
                    if (new_id[n] != -1) continue;

                    uint32_t deg = inbound_[n].size();
                    if (deg == 0) {
                        if (!active[n])
                            activate(n, INFINITY);
                    } else if (deg < min_inbound) {
                        min_inbound = deg;
                        fallback = {n};
                    } else if (deg == min_inbound) {
                        fallback.push_back(n);
                    }
                }

                if (heap.empty()) {
                    std::sort(fallback.begin(), fallback.end());
                    for (uint32_t n : fallback)
                        activate(n, INFINITY);
                }

                while (!heap.empty()) {
                    auto [p, n] = heap.top();
                    heap.pop();

                    if (new_id[n] != -1) continue;
                    if (p != priority[n]) continue;

                    new_id[n] = next_id++;
                    active[n] = false;

                    for (uint32_t hid : outbound_[n]) {
                        const HyperEdge& he = hedges_[hid];
                        const float w = he.weight();

                        for (uint32_t i = 0; i < he.length(); ++i) {
                            uint32_t m = hedges_flat_[he.offset() + i];
                            if (new_id[m] == -1)
                                activate(m, w);
                        }
                    }
                }
            }

            return new_id;
        }

        // renames nodes according to a new id give for each of them
        HyperGraph renameNodes(std::vector<int32_t> new_id) const {
            if (new_id.size() != node_count_) throw std::runtime_error("The number of new ids did not match the number of existing nodes");

            std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t>>> new_hedges;
            std::vector<float> new_weights;
            new_hedges.reserve(hedges_.size());
            new_weights.reserve(hedges_.size());

            for (const auto& he : hedges_) {
                std::pair<std::vector<uint32_t>, std::vector<uint32_t>> hv;
                hv.first.reserve(he.src_count());
                for (uint32_t s : he.sources())
                    hv.first.push_back(new_id[s]);
                hv.second.reserve(he.length() - he.src_count());
                for (uint32_t d : he.destinations())
                    hv.second.push_back(new_id[d]);
                new_hedges.push_back(std::move(hv));
                new_weights.push_back(he.weight());
            }

            return HyperGraph(node_count_, new_hedges, new_weights);
        }

        // remove duplicate nodes (and self-cycles) inside each hyperedge
        // => if a duplicate involves source + destination, what happens depends on cycle_breaking_rule:
        // 0 : keep both
        // 1 : remove the source of the cycle
        // 2 : remove the destination of the cycle
        // => when 'update_in_out_sets' is false 'inbound_' and 'outbound_' sets are NOT updated (e.g. the hedge still shows among the source's inbounds)
        void deduplicateHyperedges(uint8_t cycle_breaking_rule, bool update_in_out_sets = false) {
            std::vector<uint32_t> new_flat;
            new_flat.reserve(hedges_flat_.size());
            uint32_t new_offset = 0;
            for (auto& he : hedges_) {
                const uint32_t old_offset = he.offset();
                const uint32_t old_src_cnt = he.src_count();
                const uint32_t old_length = he.length();

                std::set<uint32_t> seen;
                uint32_t start, end;
                if (cycle_breaking_rule == 2) { // remove the destination -> insert sources first
                    start = old_offset;
                    end = old_offset + old_src_cnt;
                } else {
                    start = old_offset + old_src_cnt;
                    end = old_offset + old_length;
                }

                for (uint32_t i = start; i < end; ++i) {
                    uint32_t v = hedges_flat_[i];
                    if (seen.insert(v).second)
                        new_flat.push_back(v);
                }

                uint32_t new_halfway = static_cast<uint32_t>(new_flat.size() - new_offset);

                if (cycle_breaking_rule == 0) // keep self-cycles -> don't track duplicates
                    seen.clear();

                if (cycle_breaking_rule == 2) { // remove the destination -> now insert destinations
                    start = old_offset + old_src_cnt;
                    end = old_offset + old_length;
                } else {
                    start = old_offset;
                    end = old_offset + old_src_cnt;
                }

                for (uint32_t i = start; i < end; ++i) {
                    uint32_t v = hedges_flat_[i];
                    if (seen.insert(v).second)
                        new_flat.push_back(v);
                }

                uint32_t new_length = static_cast<uint32_t>(new_flat.size() - new_offset);
                uint32_t new_src_cnt = cycle_breaking_rule == 2 ? new_halfway : new_length - new_halfway;
                he = HyperEdge(new_offset, new_length, new_src_cnt, he.weight(), nullptr);
                new_offset += new_length;
            }

            hedges_flat_.swap(new_flat);
            hedges_flat_.shrink_to_fit();

            const uint32_t* base = hedges_flat_.data();
            for (auto& he : hedges_)
                he = HyperEdge(he.offset(), he.length(), he.src_count(), he.weight(), base);

            if (update_in_out_sets) {
                for (uint32_t i = 0; i < inbound_.size(); ++i) {
                    const auto& out = outbound_[i];
                    std::erase_if(inbound_[i], [&](auto x) { return out.contains(x); });
                }
            }
        }

        // builds all neighborhoods (in- and outbound to each node)
        void buildNeighborhoods() {
            std::cerr << "WARNING: building neighborhoods will take a while...\n";

            neighborhoods_.clear();
            neighborhood_offsets_.clear();
            neighborhood_offsets_.resize(node_count_ + 1);

            uint32_t write_pos = 0;

            for (uint32_t n = 0; n < node_count_; n++) {
                neighborhood_offsets_[n] = write_pos;

                std::set<uint32_t> neigh; // distinct neighbors

                for (uint32_t idx : outbound_[n]) {
                    const HyperEdge& he = hedges_[idx];
                    for (uint32_t i = 1; i < he.length(); i++) {
                        neigh.insert(hedges_flat_[he.offset() + i]);
                    }
                }

                for (uint32_t idx : inbound_[n]) {
                    const HyperEdge& he = hedges_[idx];
                    for (uint32_t i = 0; i < he.length(); i++) {
                        neigh.insert(hedges_flat_[he.offset() + i]);
                    }
                }

                neigh.erase(n);
                neighborhoods_.insert(neighborhoods_.end(), neigh.begin(), neigh.end());
                write_pos += neigh.size();
            }

            neighborhood_offsets_[node_count_] = write_pos;
            neighborhoods_built_ = true;
        }

        // gives the maximum neighborhood size found by sampling 'sample_size' equi-spaced nodes
        uint32_t sampleMaxNeighborhoodSize(uint32_t sample_size) {
            uint32_t max_size = 0;

            #pragma omp parallel for reduction(max:max_size) schedule(guided)
            for (uint32_t n = 0; n < node_count_; n += std::max<uint32_t>(1, node_count_ / sample_size)) {

                std::set<uint32_t> neigh; // distinct neighbors

                for (uint32_t idx : outbound_[n]) {
                    const HyperEdge& he = hedges_[idx];
                    for (uint32_t i = 1; i < he.length(); i++) {
                        neigh.insert(hedges_flat_[he.offset() + i]);
                    }
                }

                for (uint32_t idx : inbound_[n]) {
                    const HyperEdge& he = hedges_[idx];
                    for (uint32_t i = 0; i < he.length(); i++) {
                        neigh.insert(hedges_flat_[he.offset() + i]);
                    }
                }

                neigh.erase(n);

                max_size = std::max(max_size, (uint32_t)neigh.size());
                neigh.erase(n);
            }
            return max_size;
        }

        // return a view (pointer + size) of a node's neighborhood
        std::pair<const uint32_t*, size_t> getNeighborhood(Node n) const {
            if (!neighborhoods_built_)
                throw std::runtime_error("Neighborhoods not built. Call buildNeighborhoods() first.");
            if (n >= node_count_)
                throw std::runtime_error("Invalid node");

            uint32_t begin = neighborhood_offsets_[n];
            uint32_t end   = neighborhood_offsets_[n + 1];
            return { neighborhoods_.data() + begin, static_cast<size_t>(end - begin) };
        }


        // return the entire neighborhoods array
        const std::vector<uint32_t>& getNeighborhoods() const {
            if (!neighborhoods_built_)
                throw std::runtime_error("Neighborhoods not built. Call buildNeighborhoods() first.");
            return neighborhoods_;
        }

        // return the entire neighborhoods array
        const std::vector<uint32_t>& getNeighborhoodOffsets() const {
            if (!neighborhoods_built_)
                throw std::runtime_error("Neighborhoods not built. Call buildNeighborhoods() first.");
            return neighborhood_offsets_;
        }

        /*
        * SNN binary hypergraph format (.snn)
        *
        * A compact binary format for directed hypergraphs with
        * exactly one source node per hyperedge.
        *
        * File layout (little-endian):
        *
        *   uint32  node_count        // total number of nodes
        *   uint32  edge_count        // number of hyperedges
        *
        *   Repeated edge_count times:
        *     uint32  dst_count       // number of destination nodes
        *     uint32  src             // source node id (0-based)
        *     uint32  dst[dst_count]  // destination node ids (0-based)
        *     float   weight          // hyperedge weight
        *
        * Notes:
        * - Node IDs are 0-based.
        * - Hyperedges are directed: src → dst[...].
        * - The format supports exactly one source per hyperedge.
        * - If a hyperedge has multiple sources internally, only the first source
        *   is preserved; additional sources are converted into destinations and
        *   a warning is emitted during saving.
        * - No vertex weights or metadata are stored.
        */
        static HyperGraph loadSNN(const std::string& path) {
            std::ifstream f(path, std::ios::binary);
            if (!f) throw std::runtime_error("Cannot open file");

            auto R32 = [&](uint32_t& v) { f.read((char*)&v, 4); };
            auto RF = [&](float& v) { f.read((char*)&v, 4); };

            uint32_t nodes, num_edges;
            R32(nodes);
            R32(num_edges);

            std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t>>> hedges(num_edges);
            std::vector<float> weights(num_edges);

            for (uint32_t i = 0; i < num_edges; i++) {
                uint32_t dstc, src;
                R32(dstc);
                R32(src);

                std::pair<std::vector<uint32_t>, std::vector<uint32_t>> hv;
                hv.first.push_back(src);
                hv.second.reserve(dstc);

                for (uint32_t k = 0; k < dstc; k++) {
                    uint32_t d;
                    R32(d);
                    hv.second.push_back(d);
                }
                hedges[i] = hv;

                float w;
                RF(w);
                weights[i] = w;
            }

            return HyperGraph(nodes, hedges, weights);
        }

        // HP: one src per hyperedge
        void saveSNN(const std::string& path) const {
            std::ofstream f(path, std::ios::binary);
            if (!f) throw std::runtime_error("Cannot open output file");
            bool multiple_srcs_warning = false;

            uint32_t num_edges = hedges_.size();

            auto W32 = [&](uint32_t v) { f.write((char*)&v, 4); };
            auto WF = [&](float v) { f.write((char*)&v, 4); };

            W32(node_count_);
            W32(num_edges);

            for (auto& he : hedges_) {
                uint32_t dst_count = he.length() - 1;
                uint32_t src = hedges_flat_[he.offset()];
                if (he.src_count() > 1) multiple_srcs_warning = true;

                W32(dst_count);
                W32(src);

                for (uint32_t i = 1; i < he.length(); i++)
                    W32(hedges_flat_[he.offset() + i]);

                WF(he.weight());
            }

            if (multiple_srcs_warning) std::cerr << "WARNING: the partitioned hypergraph had hedge with multiple sources, a feature unsupported in '.snn' format, every source after the first was converted to a destination !!\n";
        }

        /*
         * Axon hypergraph binary format (.axh)
         *
         * A compact binary format for directed hypergraphs supporting
         * an arbitrary number of source and destination nodes per hyperedge.
         * It generalizes the SNN format while having a similar - but not compatible - layout.
         *
         * File layout (little-endian):
         *
         *   uint32  node_count        // total number of nodes
         *   uint32  edge_count        // number of hyperedges
         *
         *   Repeated edge_count times:
         *     uint32  pin_count       // total number of pins (sources + destinations)
         *     uint32  src_count       // number of source nodes (>= 1)
         *     uint32  pins[pin_count] // node ids, sources first, then destinations (0-based)
         *     float   weight          // hyperedge weight
         *
         * Semantics:
         * - pin_count >= src_count >= 1
         * - destinations are pins[src_count .. pin_count-1]
         * - Node IDs are 0-based.
         * - Hyperedges are directed from sources to destinations.
         *
         * Notes:
         * - No vertex weights or metadata are stored.
         * - This format is lossless with respect to the internal HyperGraph model.
         * - All integers and floats are stored in native little-endian format.
         */
        static HyperGraph loadAXH(const std::string& path) {
            std::ifstream f(path, std::ios::binary);
            if (!f) throw std::runtime_error("Cannot open .axh file");

            auto R32 = [&](uint32_t& v) { f.read((char*)&v, 4); };
            auto RF  = [&](float& v)    { f.read((char*)&v, 4); };

            uint32_t nodes, num_edges;
            R32(nodes);
            R32(num_edges);

            std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t>>> hedges;
            std::vector<float> weights;
            hedges.reserve(num_edges);
            weights.reserve(num_edges);

            for (uint32_t i = 0; i < num_edges; ++i) {
                uint32_t pin_count, src_count;
                R32(pin_count);
                R32(src_count);

                if (src_count == 0 || pin_count < src_count)
                    throw std::runtime_error("Invalid AXH hyperedge header");

                std::vector<uint32_t> srcs;
                std::vector<uint32_t> dsts;
                srcs.reserve(src_count);
                dsts.reserve(pin_count - src_count);

                for (uint32_t k = 0; k < pin_count; ++k) {
                    uint32_t v;
                    R32(v);
                    if (v >= nodes)
                        throw std::runtime_error("Invalid node id in AXH file");

                    if (k < src_count)
                        srcs.push_back(v);
                    else
                        dsts.push_back(v);
                }

                float w;
                RF(w);

                hedges.emplace_back(std::move(srcs), std::move(dsts));
                weights.push_back(w);
            }

            return HyperGraph(nodes, hedges, weights);
        }

        void saveAXH(const std::string& path) const {
            std::ofstream f(path, std::ios::binary);
            if (!f) throw std::runtime_error("Cannot open output .axh file");

            auto W32 = [&](uint32_t v) { f.write((char*)&v, 4); };
            auto WF  = [&](float v)    { f.write((char*)&v, 4); };

            const uint32_t num_edges = hedges_.size();

            W32(node_count_);
            W32(num_edges);

            for (const auto& he : hedges_) {
                const uint32_t pin_count = he.length();
                const uint32_t src_count = he.src_count();

                if (src_count == 0 || pin_count < src_count)
                    throw std::runtime_error("Invalid hyperedge encountered while saving AXH");

                W32(pin_count);
                W32(src_count);

                for (uint32_t i = 0; i < pin_count; ++i)
                    W32(hedges_flat_[he.offset() + i]);

                WF(he.weight());
            }
        }

        /*
        * hMETIS hypergraph format (limited support) (.hgr)
        *
        * doc: https://course.ece.cmu.edu/~ee760/760docs/hMetisManual.pdf
        *
        * This implementation supports loading from and saving to the hMETIS
        * plain-text hypergraph format as described in the hMETIS manual, with
        * the following restrictions and conventions.
        *
        * General format:
        *
        *   First non-comment line:
        *     E V [fmt]
        *
        *     E   = number of hyperedges
        *     V   = number of vertices
        *     fmt = optional format flag
        *           0  : unweighted hyperedges (default)
        *           1  : weighted hyperedges
        *          10  : weighted vertices (unsupported)
        *          11  : weighted hyperedges and vertices (vertex weights unsupported)
        *
        *   Followed by E hyperedge lines (comments starting with '%' are ignored):
        *
        *     [w] v1 v2 v3 ...
        *
        *     where:
        *       w  = integer hyperedge weight (only if fmt == 1 or 11)
        *       vi = 1-based vertex indices
        *
        * Semantics and limitations:
        *
        * - Hyperedges are treated as undirected, as per hMETIS.
        * - Vertex indices are converted from 1-based (hMETIS) to 0-based internally.
        * - Hyperedge weights are supported; unweighted hyperedges default to weight 1.
        * - Vertex weights (fmt == 10 or 11) are detected but ignored; a warning is printed.
        * - Hyperedges of cardinality 1 (singleton hyperedges) are valid in hMETIS but
        *   unsupported here; they are skipped and counted, and a warning is printed.
        *
        * Export behavior (savehMETIS):
        *
        * - Writes a valid hMETIS .hgr file using only hyperedge weights (fmt = 1 if needed).
        * - Vertex weights are never written.
        * - Hyperedges are exported by listing all pins (sources first, then destinations),
        *   preserving internal order.
        * - Node indices are written using 1-based indexing, as required by hMETIS.
        *
        * Note:
        * - Directionality and source/destination distinctions are not represented in
        *   hMETIS and are ignored on load; all pins are treated uniformly.
        */
        static HyperGraph loadhMETIS(const std::string& path) {
            std::ifstream f(path);
            if (!f) throw std::runtime_error("Cannot open .hgr file");

            auto nextDataLine = [&]() -> std::string {
                std::string line;
                while (std::getline(f, line)) {
                    if (!line.empty() && line.back() == '\r') line.pop_back();
                    size_t i = line.find_first_not_of(" \t");
                    if (i == std::string::npos) continue;
                    if (line[i] == '%') continue;
                    return line.substr(i);
                }
                return {};
            };

            std::string header = nextDataLine();
            if (header.empty()) throw std::runtime_error("Empty .hgr file");

            std::istringstream hs(header);
            uint32_t E, V;
            uint32_t fmt = 0;
            hs >> E >> V;
            if (!(hs >> fmt)) fmt = 0;

            bool edge_weights = (fmt == 1 || fmt == 11);
            bool node_weights = (fmt == 10 || fmt == 11);

            std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t>>> hedges;
            std::vector<float> weights;
            hedges.reserve(E);
            weights.reserve(E);
            uint32_t singleton_hyperedges = 0;

            for (uint32_t i = 0; i < E; ++i) {
                std::string line = nextDataLine();
                if (line.empty()) throw std::runtime_error("Unexpected end of file while reading hyperedges");

                std::istringstream ls(line);

                float w = 1.0f;
                if (edge_weights)
                    ls >> w;
                std::vector<uint32_t> nodes;
                uint32_t v;
                while (ls >> v) {
                    if (v == 0 || v > V) throw std::runtime_error("Invalid node encountered");
                    nodes.push_back(v - 1); // convert hMetis's 1-based to 0-based node id
                }
                if (nodes.size() < 1) throw std::runtime_error("Hyperedge must contain at least one node");
                else if (nodes.size() == 1) {
                    singleton_hyperedges++;
                    continue;
                }
                hedges.push_back(std::make_pair(std::vector<uint32_t>{}, std::move(nodes)));
                weights.push_back(w);
            }

            //std::vector<uint32_t> vertex_weights;
            if (node_weights) {
                std::cerr << "WARNING: .hgr files with vertex weights are not supported !!\n";
                //vertex_weights.resize(V);
                //for (uint32_t i = 0; i < V; ++i) {
                //    std::string line = nextDataLine();
                //    if (line.empty())
                //        throw std::runtime_error("Unexpected end of file while reading vertex weights");
                //    vertex_weights[i] = static_cast<uint32_t>(std::stoul(line));
                //}
            }
            if (singleton_hyperedges > 0) {
                std::cerr << "WARNING: skipped " << singleton_hyperedges << " singleton hyperedges (cardinality = 1) !!\n";
            }

            return HyperGraph(V, hedges, weights);
        }

        // HP: export undirected hyperedge, no vertex weights, only on hypredges
        void savehMETIS(const std::string& path, const float fixed_point_weights_scale = 1024.0f) const {
            std::ofstream f(path);
            if (!f) throw std::runtime_error("Cannot open output .hgr file");

            const uint32_t E = hedges_.size();
            const uint32_t V = node_count_;

            bool emit_weights = false;
            for (const auto& he : hedges_) {
                if (he.weight() != 1.0f) {
                    emit_weights = true;
                    break;
                }
            }

            // header
            if (emit_weights)
                f << E << " " << V << " 01\n";
            else
                f << E << " " << V << "\n";

            // hyperedges
            for (const auto& he : hedges_) {
                if (emit_weights)
                    f << static_cast<uint32_t>(he.weight() * fixed_point_weights_scale) << " ";

                // write all pins (sources first, then destinations)
                for (uint32_t i = 0; i < he.length(); ++i) {
                    uint32_t v = hedges_flat_[he.offset() + i];
                    f << (v + 1); // convert 0-based -> 1-based idxs
                    if (i + 1 < he.length())
                        f << " ";
                }
                f << "\n";
            }
        }
    
    };

};