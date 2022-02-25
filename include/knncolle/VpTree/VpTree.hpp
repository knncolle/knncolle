#ifndef KNNCOLLE_VPTREE_HPP
#define KNNCOLLE_VPTREE_HPP

#include "../utils/distances.hpp"
#include "../utils/NeighborQueue.hpp"
#include "../utils/Base.hpp"

#include <vector>
#include <random>
#include <limits>
#include <tuple>

/**
 * @file VpTree.hpp
 *
 * @brief Implements a vantage point (VP) tree to search for nearest neighbors.
 */

namespace knncolle {

/**
 * @brief Perform a nearest neighbor search based on a vantage point (VP) tree.
 *
 * In a vantage point tree (Yianilos, 1993), each node contains a subset of points that is split into two further partitions.
 * The split is determined by picking an arbitrary point inside that subset as the node center, 
 * computing the distance to all other points from the center, and using the median distance as the "radius" of a hypersphere.
 * The left child of this node contains all points within that hypersphere while the right child contains the remaining points.
 * This procedure is applied recursively until all points resolve to individual nodes, thus yielding a VP tree. 
 * Upon searching, the algorithm traverses the tree and exploits the triangle inequality between query points and node centers to narrow the search space.
 *
 * The major advantage of VP trees over more conventional KD-trees or ball trees is that the former does not need to construct intermediate nodes, instead using the data points themselves at the nodes.
 * This reduces the memory usage of the tree and total number of distance calculations for any search.
 * It can also be very useful when the concept of an intermediate is not well-defined (e.g., for non-numeric data), though this is not particularly relevant for **knncolle**.
 *
 * @tparam DISTANCE Class to compute the distance between vectors, see `distance::Euclidean` for an example.
 * @tparam INDEX_t Integer type for the indices.
 * @tparam DISTANCE_t Floating point type for the distances.
 * @tparam QUERY_t Floating point type for the query data.
 * @tparam INTERNAL_t Floating point type for the internal data store.
 *
 * @see
 * Yianilos PN (1993).
 * Data structures and algorithms for nearest neighbor search in general metric spaces.
 * _Proceedings of the Fourth Annual ACM-SIAM Symposium on Discrete Algorithms_, 311-321.
 * 
 * @see
 * Hanov S (2011).
 * VP trees: A data structure for finding stuff fast.
 * http://stevehanov.ca/blog/index.php?id=130
 */
template<class DISTANCE, typename INDEX_t = int, typename DISTANCE_t = double, typename QUERY_t = DISTANCE_t, typename INTERNAL_t = DISTANCE_t>
class VpTree : public Base<INDEX_t, DISTANCE_t, QUERY_t> {
    /* Adapted from http://stevehanov.ca/blog/index.php?id=130 */

private:
    INDEX_t num_dim;
    INDEX_t num_obs;
public:
    INDEX_t nobs() const { return num_obs; } 
    
    INDEX_t ndim() const { return num_dim; }
private:
    typedef int NodeIndex_t;
    static const NodeIndex_t LEAF_MARKER=-1;

    // Single node of a VP tree (has a point and radius; left children are closer to point than the radius)
    struct Node {
        INTERNAL_t threshold;  // radius 
        INDEX_t index; // original index of current vantage point 
        NodeIndex_t left;  // node index of the next vantage point for all children closer than 'threshold' from the current vantage point
        NodeIndex_t right; // node index of the next vantage point for all children further than 'threshold' from the current vantage point
        Node(NodeIndex_t i=0) : threshold(0), index(i), left(LEAF_MARKER), right(LEAF_MARKER) {}
    };
    std::vector<Node> nodes;

    typedef std::tuple<INDEX_t, const INTERNAL_t*, INTERNAL_t> DataPoint; // internal distances computed using "INTERNAL_t" type, even if output is returned with DISTANCE_t.

    template<class SAMPLER>
    NodeIndex_t buildFromPoints(NodeIndex_t lower, NodeIndex_t upper, std::vector<DataPoint>& items, SAMPLER& rng) {
        if (upper == lower) {     // indicates that we're done here!
            return LEAF_MARKER;
        }

        NodeIndex_t pos = nodes.size();
        nodes.resize(pos + 1);
        Node& node=nodes.back();
            
        int gap = upper - lower;
        if (gap > 1) {      // if we did not arrive at leaf yet

            /* Choose an arbitrary point and move it to the start of the [lower, upper)
             * interval in 'items'; this is our new vantage point.
             * 
             * Yes, I know that the modulo method does not provide strictly
             * uniform values but statistical correctness doesn't really matter
             * here... but reproducibility across platforms does matter, and
             * std::uniform_int_distribution is implementation-dependent!
             */
            NodeIndex_t i = static_cast<NodeIndex_t>(rng() % gap + lower);
            std::swap(items[lower], items[i]);
            const auto& vantage = items[lower];

            // Compute distances to the new vantage point.
            const INTERNAL_t* ref = std::get<1>(vantage);
            for (size_t i = lower + 1; i < upper; ++i) {
                const INTERNAL_t* loc = std::get<1>(items[i]);
                std::get<2>(items[i]) = DISTANCE::template raw_distance<INTERNAL_t>(ref, loc, num_dim);
            }

            // Partition around the median distance from the vantage point.
            NodeIndex_t median = lower + gap/2;
            std::nth_element(items.begin() + lower + 1, items.begin() + median, items.begin() + upper,
                [&](const DataPoint& left, const DataPoint& right) -> bool {
                    return std::get<2>(left) < std::get<2>(right);
                }
            );
           
            // Threshold of the new node will be the distance to the median
            node.threshold = DISTANCE::normalize(std::get<2>(items[median]));

            // Recursively build tree
            node.index = std::get<0>(vantage);
            node.left = buildFromPoints(lower + 1, median, items, rng);
            node.right = buildFromPoints(median, upper, items, rng);
        } else {
            node.index = std::get<0>(items[lower]);
        }
        
        return pos;
    }

private:
    std::vector<INDEX_t> new_location;
    std::vector<INTERNAL_t> store;

public:
    /**
     * @param ndim Number of dimensions.
     * @param nobs Number of observations.
     * @param vals Pointer to an array of length `ndim * nobs`, corresponding to a dimension-by-observation matrix in column-major format, 
     * i.e., contiguous elements belong to the same observation.
     *
     * @tparam INPUT_t Floating-point type of the input data.
     */
    template<typename INPUT_t>
    VpTree(INDEX_t ndim, INDEX_t nobs, const INPUT_t* vals) : num_dim(ndim), num_obs(nobs), new_location(nobs), store(ndim * nobs) { 
        std::vector<DataPoint> items;
        items.reserve(num_obs);
        for (INDEX_t i = 0; i < num_obs; ++i) {
            items.push_back(DataPoint(i, vals + i * num_dim, 0));
        }

        nodes.reserve(num_obs);
        std::mt19937_64 rand(1234567890); // seed doesn't really matter, we don't need statistical correctness here.
        buildFromPoints(0, num_obs, items, rand);

        // Actually populating the store based on the traversal order of the nodes.
        // This should be more cache efficient than an arbitrary input order.
        auto sIt = store.begin();
        for (size_t i = 0; i < num_obs; ++i, sIt += num_dim) {
            const auto& curnode = nodes[i];
            new_location[curnode.index] = i;
            auto start = vals + num_dim * curnode.index;
            std::copy(start, start + num_dim, sIt);
        }
        return;
    }

    std::vector<std::pair<INDEX_t, DISTANCE_t> > find_nearest_neighbors(INDEX_t index, int k) const {
        NeighborQueue<INDEX_t, INTERNAL_t> nearest(k, index);
        INTERNAL_t tau = std::numeric_limits<INTERNAL_t>::max();
        search_nn(0, store.data() + new_location[index] * num_dim, tau, nearest);
        return nearest.template report<DISTANCE_t>();
    }

    std::vector<std::pair<INDEX_t, DISTANCE_t> > find_nearest_neighbors(const QUERY_t* query, int k) const {
        NeighborQueue<INDEX_t, INTERNAL_t> nearest(k);
        INTERNAL_t tau = std::numeric_limits<INTERNAL_t>::max();
        search_nn(0, query, tau, nearest);
        return nearest.template report<DISTANCE_t>();
    }

    const QUERY_t* observation(INDEX_t index, QUERY_t* buffer) const {
        auto candidate = store.data() + num_dim * new_location[index];
        if constexpr(std::is_same<QUERY_t, INTERNAL_t>::value) {
            return candidate;
        } else {
            std::copy(candidate, candidate + num_dim, buffer);
            return buffer;
        }
    }

    using Base<INDEX_t, DISTANCE_t, QUERY_t>::observation;

private:
    template<typename INPUT_t>
    void search_nn(NodeIndex_t curnode_index, const INPUT_t* target, INTERNAL_t& tau, NeighborQueue<INDEX_t, INTERNAL_t>& nearest) const { 
        if (curnode_index == LEAF_MARKER) { // indicates that we're done here
            return;
        }
        
        // Compute distance between target and current node
        const auto& curnode=nodes[curnode_index];
        INTERNAL_t dist = DISTANCE::normalize(DISTANCE::template raw_distance<INTERNAL_t>(store.data() + curnode_index * num_dim, target, num_dim));

        // If current node within radius tau
        if (dist < tau) {
            nearest.add(curnode.index, dist);
            if (nearest.is_full()) {
                tau = nearest.limit(); // update value of tau (farthest point in result list)
            }
        }

        // Return if we arrived at a leaf
        if (curnode.left == LEAF_MARKER && curnode.right == LEAF_MARKER) {
            return;
        }
        
        // If the target lies within the radius of ball
        if (dist < curnode.threshold) {
            if (dist - tau <= curnode.threshold) {         // if there can still be neighbors inside the ball, recursively search left child first
                search_nn(curnode.left, target, tau, nearest);
            }
            
            if (dist + tau >= curnode.threshold) {         // if there can still be neighbors outside the ball, recursively search right child
                search_nn(curnode.right, target, tau, nearest);
            }
        
        // If the target lies outsize the radius of the ball
        } else {
            if (dist + tau >= curnode.threshold) {         // if there can still be neighbors outside the ball, recursively search right child first
                search_nn(curnode.right, target, tau, nearest);
            }
            
            if (dist - tau <= curnode.threshold) {         // if there can still be neighbors inside the ball, recursively search left child
                search_nn(curnode.left, target, tau, nearest);
            }
        }
    }
};

/**
 * Perform a VP tree search with Euclidean distances.
 */
template<typename INDEX_t = int, typename DISTANCE_t = double, typename QUERY_t = DISTANCE_t, typename INTERNAL_t = double>
using VpTreeEuclidean = VpTree<distances::Euclidean, INDEX_t, DISTANCE_t, QUERY_t, INTERNAL_t>;

/**
 * Perform a VP tree search with Manhattan distances.
 */
template<typename INDEX_t = int, typename DISTANCE_t = double, typename QUERY_t = DISTANCE_t, typename INTERNAL_t = double>
using VpTreeManhattan = VpTree<distances::Manhattan, INDEX_t, DISTANCE_t, QUERY_t, INTERNAL_t>;

};

#endif
