#ifndef KNNCOLLE_VPTREE_HPP
#define KNNCOLLE_VPTREE_HPP

#include "../utils/distances.hpp"
#include "../utils/NeighborQueue.hpp"
#include "../utils/MatrixStore.hpp"
#include "../utils/knn_base.hpp"

#include <vector>
#include <random>
#include <limits>

/**
 * @file VpTree.hpp
 *
 * Implements a vantage point (VP) tree to search for nearest neighbors.
 */


namespace knncolle {

/**
 * @brief Perform a nearest neighbor search based on a vantage point (VP) tree.
 *
 * In a VP tree (Yianilos, 1993), each node contains a subset of points that is split into two further partitions.
 * The split is determined by picking an arbitrary point inside that subset as the node center, 
 * computing the distance to all other points from the center, and using the median distance as the "radius" of a hypersphere.
 * The left child of this node contains all points within that hypersphere while the right child contains the remaining points.
 * This is applied recursively until all points resolve to individual nodes.
 *
 * The nearest neighbor search traverses the tree and exploits the triangle inequality between query points and node centers to narrow the search space.
 * VP trees are often faster than more conventional KD-trees or ball trees as the former uses the points themselves as the nodes of the tree,
 * avoiding the need to create many intermediate nodes and reducing the total number of distance calculations.
 *
 * @tparam DISTANCE Class to compute the distance between vectors, see `distance::Euclidean` for an example.
 * @tparam ITYPE Integer type for the indices.
 * @tparam DTYPE Floating point type for the data.
 *
 * @see
 * Yianilos PN (1993).
 * Data structures and algorithms for nearest neighbor search in general metric spaces.
 * _Proceedings of the Fourth Annual ACM-SIAM Symposium on Discrete Algorithms_, 311-321.
 */
template<class DISTANCE, typename ITYPE = int, typename DTYPE = double>
class VpTree : public knn_base<ITYPE, DTYPE> {
    /* Adapted from http://stevehanov.ca/blog/index.php?id=130 */

private:
    ITYPE num_dim;
    ITYPE num_obs;
public:
    ITYPE nobs() const { return num_obs; } 
    
    ITYPE ndim() const { return num_dim; }
private:
    typedef int NodeIndex_t;
    static const NodeIndex_t LEAF_MARKER=-1;

    // Single node of a VP tree (has a point and radius; left children are closer to point than the radius)
    struct Node {
        double threshold;  // radius 
        ITYPE index; // original index of current vantage point 
        NodeIndex_t left;  // node index of the next vantage point for all children closer than 'threshold' from the current vantage point
        NodeIndex_t right; // node index of the next vantage point for all children further than 'threshold' from the current vantage point
        Node(NodeIndex_t i=0) : threshold(0), index(i), left(LEAF_MARKER), right(LEAF_MARKER) {}
    };
    std::vector<Node> nodes;

    typedef std::tuple<ITYPE, const double*, double> DataPoint;

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
            const double * ref = std::get<1>(vantage);
            for (size_t i = lower + 1; i < upper; ++i) {
                const double* loc = std::get<1>(items[i]);
                std::get<2>(items[i]) = DISTANCE::raw_distance(ref, loc, num_dim);
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
    MatrixStore<DTYPE> store;

public:
    /**
     * Construct a `VpTree` instance without any copying of the data.
     * The `vals` pointer is directly stored in the instance, assuming that the lifetime of the array exceeds that of the `BruteForce` object.
     *
     * @param ndim Number of dimensions.
     * @param nobs Number of observations.
     * @param vals Pointer to an array of length `ndim * nobs`, corresponding to a dimension-by-observation matrix in column-major format, 
     * i.e., contiguous elements belong to the same observation.
     */
    VpTree(ITYPE ndim, ITYPE nobs, const DTYPE* vals) : num_dim(ndim), num_obs(nobs), store(vals) { 
        complete_assembly();
        return;
    }

    /**
     * Construct a `VpTree` instance by copying the data.
     * This is useful when the original data container has an unknown lifetime.
     *
     * @param ndim Number of dimensions.
     * @param nobs Number of observations.
     * @param vals Vector of length `ndim * nobs`, corresponding to a dimension-by-observation matrix in column-major format, 
     * i.e., contiguous elements belong to the same observation.
     */
    VpTree(ITYPE ndim, ITYPE nobs, std::vector<DTYPE> vals) : num_dim(ndim), num_obs(nobs), store(std::move(vals)) { 
        complete_assembly();
        return;
    }

private:
    void complete_assembly() {
        std::vector<DataPoint> items;
        items.reserve(num_obs);
        auto copy = store.reference;
        for (ITYPE i = 0; i < num_obs; ++i, copy += num_dim) {
            items.push_back(DataPoint(i, copy, 0));
        }

        nodes.reserve(num_obs);
        std::mt19937_64 rand(1234567890); // seed doesn't really matter, we don't need statistical correctness here.
        buildFromPoints(0, num_obs, items, rand);
        return;
    }

public:
    void find_nearest_neighbors(ITYPE index, int k, std::vector<ITYPE>* indices, std::vector<DTYPE>* distances) const { 
        assert(index < num_obs);
        NeighborQueue<ITYPE, DTYPE> nearest(k + 1);
        double tau = std::numeric_limits<double>::max();
        search_nn(0, store.reference + index * num_dim, tau, nearest);
        nearest.report(indices, distances, true, index);
        return;
    }

    void find_nearest_neighbors(const DTYPE* query, int k, std::vector<ITYPE>* indices, std::vector<DTYPE>* distances) const {
        NeighborQueue<ITYPE, DTYPE> nearest(k);
        double tau = std::numeric_limits<double>::max();
        search_nn(0, query, tau, nearest);
        nearest.report(indices, distances);
        return;
    }

private:
    void search_nn(NodeIndex_t curnode_index, const double* target, double& tau, NeighborQueue<ITYPE, DTYPE>& nearest) const { 
        if (curnode_index == LEAF_MARKER) { // indicates that we're done here
            return;
        }
        
        // Compute distance between target and current node
        const auto& curnode=nodes[curnode_index];
        double dist = DISTANCE::normalize(DISTANCE::raw_distance(store.reference + curnode.index * num_dim, target, num_dim));

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
template<typename ITYPE = int, typename DTYPE = double>
using VpTreeEuclidean = VpTree<distances::Euclidean, ITYPE, DTYPE>;

/**
 * Perform a VP tree search with Manhattan distances.
 */
template<typename ITYPE = int, typename DTYPE = double>
using VpTreeManhattan = VpTree<distances::Manhattan, ITYPE, DTYPE>;

};

#endif
