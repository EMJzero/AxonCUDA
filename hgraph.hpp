#pragma once
#include <set>
#include <vector>
#include <ranges>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <functional>
#include <unordered_map>

namespace hgraph {

    class HyperEdge;
    class HyperGraph;

    /*
    * HyperEdge:
    * - offset into the HyperGraph's node array
    * - length
    * - weight
    */
    class HyperEdge {
        public:
        using Node = uint32_t;

        private:
        uint32_t offset_; // offset inside HyperGraph::hedges_flat_
        uint32_t length_; // total nodes = 1 + destinations
        float weight_;
        const uint32_t* hedges_flat_; // pointer to the owning hypergraph's hedges_flat_ array

        public:
        HyperEdge(uint32_t offset, uint32_t length, float weight, const uint32_t* hedges_flat) : offset_(offset), length_(length), weight_(weight), hedges_flat_(hedges_flat) {}

        uint32_t offset() const { return offset_; }
        uint32_t length() const { return length_; }
        float weight() const { return weight_; }
        // aliases for compatibility
        float spikeFrequency() const { return weight_; }
        uint32_t size() const { return length_; }
        uint32_t connections() const { return length_ - 1; }

        void updateWeight(float w) { weight_ = w; }

        // These are "views", the real data lives in HyperGraph.
        // HyperGraph must provide access to hedges_flat_,
        // therefore HyperGraph declares HyperEdge as a "friend".
        const uint32_t& source() const {
            return hedges_flat_[offset_];
        }
        std::vector<uint32_t> destinations() const {
            return std::vector<uint32_t>(hedges_flat_ + offset_ + 1, hedges_flat_ + offset_ + length_);
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
        // - hedges: vector of vectors each vector must be: [src, dst1, dst2, ...]
        // - weights: same length as hedges, one weight per hyperedge.
        HyperGraph(uint32_t nodes, const std::vector<std::vector<uint32_t>>& hedges, const std::vector<float>& weights) : node_count_(nodes) {
            if (hedges.size() != weights.size())
                throw std::runtime_error("Number of hyperedges != number of weights");

            // compute total nodes needed
            size_t total_pins = 0;
            for (auto& he : hedges) {
                if (he.size() < 2)
                    throw std::runtime_error("Hyperedge must have at least 1 source and 1 destination");
                total_pins += he.size();
            }

            hedges_flat_.reserve(total_pins);
            hedges_.reserve(hedges.size());

            // build edges and contiguous array
            uint32_t offset = 0;
            for (size_t i = 0; i < hedges.size(); i++) {
                const auto& he = hedges[i];
                float w = weights[i];

                for (uint32_t v : he)
                    hedges_flat_.push_back(v);

                hedges_.emplace_back(offset, he.size(), w, hedges_flat_.data());
                offset += he.size();
            }

            // build inbound/outbound maps
            outbound_.resize(nodes);
            inbound_.resize(nodes);

            for (uint32_t idx = 0; idx < hedges_.size(); idx++) {
                auto& he = hedges_[idx];
                uint32_t src = he.source();

                if (src >= nodes)
                    throw std::runtime_error("Invalid source node");

                outbound_[src].insert(idx);

                for (uint32_t dst : he.destinations()) {
                    if (dst >= nodes)
                        throw std::runtime_error("Invalid destination node");

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
        float totalSpikeFrequency() const { return totalWeight(); }

        HyperGraph getPartitionsHypergraph(const std::vector<uint32_t>& part, bool keep_self_cycles, bool squish_hyperedges) const {
            if (part.size() != node_count_)
                throw std::runtime_error("Partition size mismatch");

            uint32_t new_nodes = 0;
            for (uint32_t p : part) new_nodes = std::max(new_nodes, p + 1);

            for (uint32_t i = 0; i < new_nodes; i++) {
                if (std::find(part.begin(), part.end(), i) == part.end())
                    throw std::runtime_error("Partitions must be 0..N sequential");
            }

            std::vector<std::vector<uint32_t>> new_hedges;
            std::vector<float> new_weights;

            if (!squish_hyperedges) {
                for (auto& he : hedges_) {
                    uint32_t srcp = part[he.source()];
                    std::set<uint32_t> dstp;

                    for (uint32_t i = 1; i < he.length(); i++)
                        dstp.insert(part[hedges_flat_[he.offset() + i]]);

                    if (!keep_self_cycles)
                        dstp.erase(srcp);

                    if (!dstp.empty()) {
                        std::vector<uint32_t> hv;
                        hv.push_back(srcp);
                        hv.insert(hv.end(), dstp.begin(), dstp.end());
                        new_hedges.push_back(hv);
                        new_weights.push_back(he.weight());
                    }
                }
            } else {
                std::unordered_map<uint64_t, float> freq_acc;
                std::unordered_map<uint64_t, std::vector<uint32_t>> dst_map;
                std::unordered_map<uint64_t, uint32_t> src_map;

                for (auto& he : hedges_) {
                    uint32_t srcp = part[he.source()];
                    std::set<uint32_t> dstp;

                    for (uint32_t i = 1; i < he.length(); i++)
                        dstp.insert(part[hedges_flat_[he.offset() + i]]);

                    if (!keep_self_cycles)
                        dstp.erase(srcp);

                    if (dstp.empty()) continue;

                    uint64_t key = 146527;
                    key ^= std::hash<uint32_t>()(srcp) + 0x9e3779b97f4a7c15ULL;
                    for (auto d : dstp)
                        key ^= std::hash<uint32_t>()(d) + 0x9e3779b97f4a7c15ULL + (key << 6) + (key >> 2);

                    freq_acc[key] += he.weight();
                    src_map[key] = srcp;
                    dst_map[key] = std::vector<uint32_t>(dstp.begin(), dstp.end());
                }

                for (auto& it : freq_acc) {
                    uint64_t k = it.first;
                    float w = it.second;

                    std::vector<uint32_t> hv;
                    hv.push_back(src_map[k]);
                    auto& dsts = dst_map[k];
                    hv.insert(hv.end(), dsts.begin(), dsts.end());
                    new_hedges.push_back(hv);
                    new_weights.push_back(w);
                }
            }

            return HyperGraph(new_nodes, new_hedges, new_weights);
        }

        // builds all neighborhoods (in- and outbound to each node)
        void buildNeighborhoods() {
            std::cout << "WARNING: building neighborhoods will take a while...\n";

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
                    neigh.insert(he.source());
                    for (uint32_t i = 1; i < he.length(); i++) {
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

            for (uint32_t n = 0; n < node_count_; n += node_count_ / sample_size) {

                std::set<uint32_t> neigh; // distinct neighbors

                for (uint32_t idx : outbound_[n]) {
                    const HyperEdge& he = hedges_[idx];
                    for (uint32_t i = 1; i < he.length(); i++) {
                        neigh.insert(hedges_flat_[he.offset() + i]);
                    }
                }

                for (uint32_t idx : inbound_[n]) {
                    const HyperEdge& he = hedges_[idx];
                    neigh.insert(he.source());
                    for (uint32_t i = 1; i < he.length(); i++) {
                        neigh.insert(hedges_flat_[he.offset() + i]);
                    }
                }

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

        void save(const std::string& path) const {
            std::ofstream f(path, std::ios::binary);
            if (!f) throw std::runtime_error("Cannot open output file");

            uint32_t num_edges = hedges_.size();

            auto W32 = [&](uint32_t v) { f.write((char*)&v, 4); };
            auto WF  = [&](float v)    { f.write((char*)&v, 4); };

            W32(node_count_);
            W32(num_edges);

            for (auto& he : hedges_) {
                uint32_t dst_count = he.length() - 1;
                uint32_t src = hedges_flat_[he.offset()];

                W32(dst_count);
                W32(src);

                for (uint32_t i = 1; i < he.length(); i++)
                    W32(hedges_flat_[he.offset() + i]);

                WF(he.weight());
            }
        }

        static HyperGraph load(const std::string& path) {
            std::ifstream f(path, std::ios::binary);
            if (!f) throw std::runtime_error("Cannot open file");

            auto R32 = [&](uint32_t& v) { f.read((char*)&v, 4); };
            auto RF  = [&](float& v)    { f.read((char*)&v, 4); };

            uint32_t nodes, num_edges;
            R32(nodes);
            R32(num_edges);

            std::vector<std::vector<uint32_t>> hedges(num_edges);
            std::vector<float> weights(num_edges);

            for (uint32_t i = 0; i < num_edges; i++) {
                uint32_t dstc, src;
                R32(dstc);
                R32(src);

                std::vector<uint32_t> hv;
                hv.reserve(dstc + 1);
                hv.push_back(src);

                for (uint32_t k = 0; k < dstc; k++) {
                    uint32_t d;
                    R32(d);
                    hv.push_back(d);
                }
                hedges[i] = hv;

                float w;
                RF(w);
                weights[i] = w;
            }

            return HyperGraph(nodes, hedges, weights);
        }
    };
};