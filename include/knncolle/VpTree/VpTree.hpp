#ifndef KNNCOLLE_VPTREE_HPP
#define KNNCOLLE_VPTREE_HPP

#include "../utils/distances.hpp"
#include "../utils/NeighborQueue.hpp"
#include "../utils/MatrixStore.hpp"
#include "../utils/knn_base.hpp"

#include <vector>
#include <random>
#include <limits>

namespace knncolle {

/* Adapted from http://stevehanov.ca/blog/index.php?id=130 */

template<bool COPY, class DISTANCE>
class VpTree : public knn_base {
private:
    MatDim_t num_dim;
    CellIndex_t num_obs;
public:
    CellIndex_t nobs() const { return num_obs; } 
    
    MatDim_t ndims() const { return num_dim; }

private:
    typedef int NodeIndex_t;
    static const NodeIndex_t LEAF_MARKER=-1;

    // Single node of a VP tree (has a point and radius; left children are closer to point than the radius)
    struct Node {
        double threshold;  // radius 
        CellIndex_t index; // original index of current vantage point 
        NodeIndex_t left;  // node index of the next vantage point for all children closer than 'threshold' from the current vantage point
        NodeIndex_t right; // node index of the next vantage point for all children further than 'threshold' from the current vantage point
        Node(NodeIndex_t i=0) : threshold(0), index(i), left(LEAF_MARKER), right(LEAF_MARKER) {}
    };
    std::vector<Node> nodes;

    typedef std::tuple<CellIndex_t, const double*, double> DataPoint;

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
    MatrixStore<COPY> store;

public:
    VpTree(CellIndex_t nobs, MatDim_t ndim, const double* vals) : num_dim(ndim), num_obs(nobs), store(ndim * nobs, vals) { 
        std::vector<DataPoint> items;
        items.reserve(nobs);
        auto copy = store.reference;
        for (MatDim_t i = 0; i < nobs; ++i, copy += ndim) {
            items.push_back(DataPoint(i, copy, 0));
        }

        nodes.reserve(nobs);
        std::mt19937_64 rand(1234567890); // seed doesn't really matter, we don't need statistical correctness here.
        buildFromPoints(0, nobs, items, rand);
        return;
    }

public:
    void find_nearest_neighbors(CellIndex_t index, NumNeighbors_t k) {
        assert(index < static_cast<CellIndex_t>(num_obs));
        NeighborQueue nearest(index, k, this->get_ties);
        find_nearest_neighbors_internal(store.reference + index * num_dim, nearest);
        return;
    }

    void find_nearest_neighbors(const double* query, NumNeighbors_t k) {
        NeighborQueue nearest(k, this->get_ties);
        find_nearest_neighbors_internal(query, nearest);
        return;
    }

private:
    void find_nearest_neighbors_internal(const double* query, NeighborQueue& nearest) {
        double tau = std::numeric_limits<double>::max();
        search_nn(0, query, tau, nearest);
        nearest.report(this->current_neighbors, this->current_distances, this->current_tied, this->get_index, this->get_distance);
        return;
    }

private:
    void search_nn(NodeIndex_t curnode_index, const double* target, double& tau, NeighborQueue& nearest) { 
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

template<bool COPY = false>
using VpTreeEuclidean = VpTree<COPY, distances::Euclidean>;

template<bool COPY = false>
using VpTreeManhattan = VpTree<COPY, distances::Manhattan>;

};

#endif
