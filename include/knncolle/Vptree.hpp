#ifndef KNNCOLLE_VPTREE_HPP
#define KNNCOLLE_VPTREE_HPP

#include "distances.hpp"
#include "NeighborQueue.hpp"
#include "Prebuilt.hpp"
#include "Builder.hpp"
#include "MockMatrix.hpp"
#include "report_all_neighbors.hpp"

#include <vector>
#include <random>
#include <limits>
#include <tuple>
#include <memory>

/**
 * @file Vptree.hpp
 *
 * @brief Implements a vantage point (VP) tree to search for nearest neighbors.
 */

namespace knncolle {

/**
 * @cond
 */
template<typename Dim_, typename Index_, typename Data_, typename Distance_, class DistanceMetric_, typename Store_>
class VptreePrebuilt;
/**
 * @endcond
 */

/**
 * @brief VP-tree searcher.
 *
 * Instances of this class are usually constructed using `VptreePrebuilt::initialize()`.
 *
 * @tparam DistanceMetric_ A distance calculation class satisfying the `MockDistance` contract.
 * @tparam Dim_ Integer type for the number of dimensions.
 * @tparam Index_ Integer type for the indices.
 * @tparam Data_ Numeric type for the input and query data.
 * @tparam Distance_ Floating point type for the distances.
 * @tparam Store_ Numeric type for the stored data.
 * This may be a lower-precision type than `Data_` to reduce memory usage.
 */
template<typename Dim_, typename Index_, typename Data_, typename Distance_, class DistanceMetric_, typename Store_>
class VptreeSearcher final : public Searcher<Index_, Data_, Distance_> {
public:
    /**
     * @cond
     */
    VptreeSearcher(const VptreePrebuilt<Dim_, Index_, Data_, Distance_, DistanceMetric_, Store_>& parent) : my_parent(parent) {}
    /**
     * @endcond
     */

private:                
    const VptreePrebuilt<Dim_, Index_, Data_, Distance_, DistanceMetric_, Store_>& my_parent;
    internal::NeighborQueue<Index_, Distance_> my_nearest;
    std::vector<std::pair<Distance_, Index_> > my_all_neighbors;

public:
    void search(Index_ i, Index_ k, std::vector<Index_>* output_indices, std::vector<Distance_>* output_distances) {
        my_nearest.reset(k + 1);
        auto iptr = my_parent->my_data.data() + static_cast<size_t>(my_parent->my_new_locations[i]) * my_parent->my_long_ndim; // cast to avoid overflow.
        Distance_ max_dist = std::numeric_limits<Distance_>::max();
        my_parent->search_nn(0, iptr, max_dist, my_nearest);
        my_nearest.report(output_indices, output_distances, i);
    }

    void search(const Data_* query, Index_ k, std::vector<Index_>* output_indices, std::vector<Distance_>* output_distances) {
        if (k == 0) { // protect the NeighborQueue from k = 0.
            internal::flush_output(output_indices, output_distances, 0);
        } else {
            my_nearest.reset(k);
            Distance_ max_dist = std::numeric_limits<Distance_>::max();
            my_parent->search_nn(0, query, max_dist, my_nearest);
            my_nearest.report(output_indices, output_distances);
        }
    }

    bool can_search_all() const {
        return true;
    }

    Index_ search_all(Index_ i, Distance_ d, std::vector<Index_>* output_indices, std::vector<Distance_>* output_distances) {
        auto iptr = my_parent->my_data.data() + static_cast<size_t>(my_parent->my_new_locations[i]) * my_parent->my_long_ndim; // cast to avoid overflow.

        if (!output_indices && !output_distances) {
            Index_ count = 0;
            my_parent->template search_all<true>(0, iptr, d, count);
            return internal::safe_remove_self(count);

        } else {
            my_all_neighbors.clear();
            my_parent->template search_all<false>(0, iptr, d, my_all_neighbors);
            internal::report_all_neighbors(my_all_neighbors, output_indices, output_distances, i);
            return internal::safe_remove_self(my_all_neighbors.size());
        }
    }

    Index_ search_all(const Data_* query, Distance_ d, std::vector<Index_>* output_indices, std::vector<Distance_>* output_distances) {
        if (!output_indices && !output_distances) {
            Index_ count = 0;
            my_parent->template search_all<true>(0, query, d, count);
            return count;

        } else {
            my_all_neighbors.clear();
            my_parent->template search_all<false>(0, query, d, my_all_neighbors);
            internal::report_all_neighbors(my_all_neighbors, output_indices, output_distances);
            return my_all_neighbors.size();
        }
    }
};

/**
 * @brief Index for a VP-tree search.
 *
 * Instances of this class are usually constructed using `VptreeBuilder::build_raw()`.
 *
 * @tparam DistanceMetric_ A distance calculation class satisfying the `MockDistance` contract.
 * @tparam Dim_ Integer type for the number of dimensions.
 * @tparam Index_ Integer type for the indices.
 * @tparam Data_ Numeric type for the input and query data.
 * @tparam Distance_ Floating point type for the distances.
 * @tparam Store_ Numeric type for the stored data.
 * This may be a lower-precision type than `Data_` to reduce memory usage.
 */
template<typename Dim_, typename Index_, typename Data_, typename Distance_, class DistanceMetric_, typename Store_>
class VptreePrebuilt final : public Prebuilt<Dim_, Index_, Data_, Distance_> {
private:
    Dim_ my_dim;
    Index_ my_obs;
    size_t my_long_ndim;
    std::vector<Store_> my_data;
    std::shared_ptr<const DistanceMetric_> my_metric;

public:
    Index_ num_observations() const {
        return my_obs;
    } 

    Dim_ num_dimensions() const {
        return my_dim;
    }

private:
    /* Adapted from http://stevehanov.ca/blog/index.php?id=130 */
    static const Index_ LEAF = 0;

    // Single node of a VP tree. 
    struct Node {
        Distance_ radius = 0;  

        // Original index of current vantage point, defining the center of the node.
        Index_ index = 0;

        // Node index of the next vantage point for all children closer than 'threshold' from the current vantage point.
        // This must be > 0, as the first node in 'nodes' is the root and cannot be referenced from other nodes.
        Index_ left = LEAF;  

        // Node index of the next vantage point for all children further than 'threshold' from the current vantage point.
        // This must be > 0, as the first node in 'nodes' is the root and cannot be referenced from other nodes.
        Index_ right = LEAF; 
    };

    std::vector<Node> my_nodes;

    typedef std::pair<Distance_, Index_> DataPoint; 

    template<class Rng_>
    Index_ build(Index_ lower, Index_ upper, const Store_* coords, std::vector<DataPoint>& items, Rng_& rng) {
        /* 
         * We're assuming that lower < upper at each point within this
         * recursion. This requires some protection at the call site
         * when nobs = 0, see the constructor.
         */

        Index_ pos = my_nodes.size();
        my_nodes.emplace_back();
        Node& node = my_nodes.back(); // this is safe during recursion because we reserved 'nodes' already to the number of observations, see the constructor.

        Index_ gap = upper - lower;
        if (gap > 1) { // not yet at a leaft.

            /* Choose an arbitrary point and move it to the start of the [lower, upper)
             * interval in 'items'; this is our new vantage point.
             * 
             * Yes, I know that the modulo method does not provide strictly
             * uniform values but statistical correctness doesn't really matter
             * here, and I don't want std::uniform_int_distribution's
             * implementation-specific behavior.
             */
            Index_ i = (rng() % gap + lower);
            std::swap(items[lower], items[i]);
            const auto& vantage = items[lower];
            node.index = vantage.second;
            const Store_* vantage_ptr = coords + static_cast<size_t>(vantage.second) * my_long_ndim; // cast to avoid overflow.

            // Compute distances to the new vantage point.
            for (Index_ i = lower + 1; i < upper; ++i) {
                const Store_* loc = coords + static_cast<size_t>(items[i].second) * my_long_ndim; // cast to avoid overflow.
                items[i].first = my_metric->raw_distance(vantage_ptr, loc, my_dim);
            }

            // Partition around the median distance from the vantage point.
            Index_ median = lower + gap/2;
            Index_ lower_p1 = lower + 1; // excluding the vantage point itself, obviously.
            std::nth_element(items.begin() + lower_p1, items.begin() + median, items.begin() + upper);

            // Radius of the new node will be the distance to the median.
            node.radius = my_metric->normalize(items[median].first);

            // Recursively build tree.
            if (lower_p1 < median) {
                node.left = build(lower_p1, median, coords, items, rng);
            }
            if (median < upper) {
                node.right = build(median, upper, coords, items, rng);
            }

        } else {
            const auto& leaf = items[lower];
            node.index = leaf.second;
        }

        return pos;
    }

private:
    std::vector<Index_> my_new_locations;

public:
    /**
     * @param num_dim Number of dimensions.
     * @param num_obs Number of observations.
     * @param data Vector of length equal to `num_dim * num_obs`, containing a column-major matrix where rows are dimensions and columns are observations.
     */
    VptreePrebuilt(Dim_ num_dim, Index_ num_obs, std::vector<Store_> data, std::shared_ptr<const DistanceMetric_> metric) : 
        my_dim(num_dim),
        my_obs(num_obs),
        my_long_ndim(my_dim),
        my_data(std::move(data)),
        my_metric(std::move(metric))
    {
        if (num_obs) {
            std::vector<DataPoint> items;
            items.reserve(my_obs);
            for (Index_ i = 0; i < my_obs; ++i) {
                items.emplace_back(0, i);
            }

            my_nodes.reserve(my_obs);

            // Statistical correctness doesn't matter (aside from tie breaking)
            // so we'll just use a deterministically 'random' number to ensure
            // we get the same ties for any given dataset but a different stream
            // of numbers between datasets. Casting to get well-defined overflow. 
            uint64_t base = 1234567890, m1 = my_obs, m2 = my_dim;
            std::mt19937_64 rand(base * m1 +  m2);

            build(0, my_obs, my_data.data(), items, rand);

            // Resorting data in place to match order of occurrence within
            // 'nodes', for better cache locality.
            std::vector<uint8_t> used(my_obs);
            std::vector<Store_> buffer(my_dim);
            my_new_locations.resize(my_obs);
            auto host = my_data.data();

            for (Index_ o = 0; o < num_obs; ++o) {
                if (used[o]) {
                    continue;
                }

                auto& current = my_nodes[o];
                my_new_locations[current.index] = o;
                if (current.index == o) {
                    continue;
                }

                auto optr = host + static_cast<size_t>(o) * my_long_ndim;
                std::copy_n(optr, my_dim, buffer.begin());
                Index_ replacement = current.index;

                do {
                    auto rptr = host + static_cast<size_t>(replacement) * my_long_ndim;
                    std::copy_n(rptr, my_dim, optr);
                    used[replacement] = 1;

                    const auto& next = my_nodes[replacement];
                    my_new_locations[next.index] = replacement;

                    optr = rptr;
                    replacement = next.index;
                } while (replacement != o);

                std::copy(buffer.begin(), buffer.end(), optr);
            }
        }
    }

private:
    void search_nn(Index_ curnode_index, const Data_* target, Distance_& max_dist, internal::NeighborQueue<Index_, Distance_>& nearest) const { 
        auto nptr = my_data.data() + static_cast<size_t>(curnode_index) * my_long_ndim; // cast to avoid overflow.
        Distance_ dist = my_metric->normalize(my_metric->raw_distance(my_dim, nptr, target));

        // If current node is within the maximum distance:
        const auto& curnode = my_nodes[curnode_index];
        if (dist <= max_dist) {
            nearest.add(curnode.index, dist);
            if (nearest.is_full()) {
                max_dist = nearest.limit(); // update value of max_dist (farthest point in result list)
            }
        }

        if (dist < curnode.radius) { // If the target lies within the radius of ball:
            if (curnode.left != LEAF && dist - max_dist <= curnode.radius) { // if there can still be neighbors inside the ball, recursively search left child first
                search_nn(curnode.left, target, max_dist, nearest);
            }

            if (curnode.right != LEAF && dist + max_dist >= curnode.radius) { // if there can still be neighbors outside the ball, recursively search right child
                search_nn(curnode.right, target, max_dist, nearest);
            }

        } else { // If the target lies outsize the radius of the ball:
            if (curnode.right != LEAF && dist + max_dist >= curnode.radius) { // if there can still be neighbors outside the ball, recursively search right child first
                search_nn(curnode.right, target, max_dist, nearest);
            }

            if (curnode.left != LEAF && dist - max_dist <= curnode.radius) { // if there can still be neighbors inside the ball, recursively search left child
                search_nn(curnode.left, target, max_dist, nearest);
            }
        }
    }

    template<bool count_only_, typename Output_>
    void search_all(Index_ curnode_index, const Data_* target, Distance_ threshold, Output_& all_neighbors) const { 
        auto nptr = my_data.data() + static_cast<size_t>(curnode_index) * my_long_ndim; // cast to avoid overflow.
        Distance_ dist = my_metric->normalize(my_metric->raw_distance(my_dim, nptr, target));

        // If current node is within the maximum distance:
        const auto& curnode = my_nodes[curnode_index];
        if (dist <= threshold) {
            if constexpr(count_only_) {
                ++all_neighbors;
            } else {
                all_neighbors.emplace_back(dist, curnode.index);
            }
        }

        if (dist < curnode.radius) { // If the target lies within the radius of ball:
            if (curnode.left != LEAF && dist - threshold <= curnode.radius) { // if there can still be neighbors inside the ball, recursively search left child first
                search_all<count_only_>(curnode.left, target, threshold, all_neighbors);
            }

            if (curnode.right != LEAF && dist + threshold >= curnode.radius) { // if there can still be neighbors outside the ball, recursively search right child
                search_all<count_only_>(curnode.right, target, threshold, all_neighbors);
            }

        } else { // If the target lies outsize the radius of the ball:
            if (curnode.right != LEAF && dist + threshold >= curnode.radius) { // if there can still be neighbors outside the ball, recursively search right child first
                search_all<count_only_>(curnode.right, target, threshold, all_neighbors);
            }

            if (curnode.left != LEAF && dist - threshold <= curnode.radius) { // if there can still be neighbors inside the ball, recursively search left child
                search_all<count_only_>(curnode.left, target, threshold, all_neighbors);
            }
        }
    }

    friend class VptreeSearcher<Dim_, Index_, Data_, Distance_, DistanceMetric_, Store_>;

public:
    /**
     * Creates a `VptreeSearcher` instance.
     */
    std::unique_ptr<Searcher<Index_, Distance_> > initialize() const {
        return std::make_unique<VptreeSearcher<Dim_, Index_, Data_, Distance_, DistanceMetric_, Store_> >(this);
    }
};

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
 * @tparam DistanceMetric_ A distance calculation class satisfying the `MockDistance` contract.
 * @tparam Dim_ Integer type for the number of dimensions.
 * @tparam Index_ Integer type for the indices.
 * @tparam Data_ Numeric type for the input and query data.
 * @tparam Distance_ Floating point type for the distances.
 * @tparam Store_ Numeric type for the stored data.
 * This may be a lower-precision type than `Data_` to reduce memory usage.
 * @tparam Matrix_ Class that satisfies the `Matrix` interface.
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
template<
    typename Dim_,
    typename Index_,
    typename Data_,
    typename Distance_,
    class DistanceMetric_ = DistanceMetric<Dim_, Data_, Distance_>,
    typename Store_ = Data_ = Data_,
    class Matrix_ = SimpleMatrix<Dim_, Index_, Data_>
>
class VptreeBuilder final : public Builder<Dim_, Index_, Data_, Distance_, Matrix_> {
public:
    /**
     * @param metric Pointer to a distance metric instance, e.g., `EuclideanDistance`.
     */
    VptreeForceBuilder(std::shared_ptr<const DistanceMetric_> metric) : my_metric(std::move(metric)) {}

    /**
     * @param metric Pointer to a distance metric instance, e.g., `EuclideanDistance`.
     */
    VptreeForceBuilder(const DistanceMetric_* metric) : VptreeForceBuilder(std::shared_ptr<const DistanceMetric_>(metric)) {}

private:
    std::shared_ptr<const DistanceMetric_> my_metric;

public:
    /**
     * Creates a `VptreePrebuilt` instance.
     */
    Prebuilt<Dim_, Index_, Data_, Distance_>* build_raw(const Matrix_& data) const {
        size_t ndim = data.num_dimensions();
        size_t nobs = data.num_observations();
        auto work = data.new_extractor();

        std::vector<Store_> store(ndim * nobs);
        for (size_t o = 0; o < nobs; ++o) {
            std::copy_n(work->next(), ndim, store.begin() + o * ndim);
        }

        return new VptreePrebuilt<Dim_, Index_, Data_, Distance_, DistanceMetric_, Store_>(ndim, nobs, std::move(store), my_metric);
    }
};

};

#endif
