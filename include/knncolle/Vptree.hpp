#ifndef KNNCOLLE_VPTREE_HPP
#define KNNCOLLE_VPTREE_HPP

#include "distances.hpp"
#include "NeighborQueue.hpp"
#include "Prebuilt.hpp"
#include "Builder.hpp"
#include "Matrix.hpp"
#include "report_all_neighbors.hpp"
#include "utils.hpp"

#include <vector>
#include <random>
#include <limits>
#include <tuple>
#include <memory>
#include <cstddef>
#include <string>
#include <cstring>
#include <filesystem>

#include "sanisizer/sanisizer.hpp"

/**
 * @file Vptree.hpp
 *
 * @brief Implements a vantage point (VP) tree to search for nearest neighbors.
 */

namespace knncolle {

/**
 * Name of the VP-tree algorithm when registering a loading function to `load_prebuilt_registry()`.
 */
inline static constexpr const char* vptree_prebuilt_save_name = "knncolle::Vptree";

/**
 * @cond
 */
template<typename Index_, typename Data_, typename Distance_, class DistanceMetric_>
class VptreePrebuilt;

template<typename Index_, typename Data_, typename Distance_, class DistanceMetric_>
class VptreeSearcher final : public Searcher<Index_, Data_, Distance_> {
public:
    VptreeSearcher(const VptreePrebuilt<Index_, Data_, Distance_, DistanceMetric_>& parent) : my_parent(parent) {}

private:                
    const VptreePrebuilt<Index_, Data_, Distance_, DistanceMetric_>& my_parent;
    NeighborQueue<Index_, Distance_> my_nearest;
    std::vector<std::pair<Distance_, Index_> > my_all_neighbors;

public:
    void search(Index_ i, Index_ k, std::vector<Index_>* output_indices, std::vector<Distance_>* output_distances) {
        my_nearest.reset(k + 1); // +1 is safe as k < num_obs.
        auto iptr = my_parent.my_data.data() + sanisizer::product_unsafe<std::size_t>(my_parent.my_new_locations[i], my_parent.my_dim);
        Distance_ max_dist = std::numeric_limits<Distance_>::max();
        my_parent.search_nn(0, iptr, max_dist, my_nearest);
        my_nearest.report(output_indices, output_distances, i);
    }

    void search(const Data_* query, Index_ k, std::vector<Index_>* output_indices, std::vector<Distance_>* output_distances) {
        // Protect the NeighborQueue from k = 0. This also protects search_nn()
        // when there are no observations (and no node 0 to start recursion). 
        if (k == 0 || my_parent.my_nodes.empty()) {
            if (output_indices) {
                output_indices->clear();
            }
            if (output_distances) {
                output_distances->clear();
            }

        } else {
            my_nearest.reset(k);
            Distance_ max_dist = std::numeric_limits<Distance_>::max();
            my_parent.search_nn(0, query, max_dist, my_nearest);
            my_nearest.report(output_indices, output_distances);
        }
    }

    bool can_search_all() const {
        return true;
    }

    Index_ search_all(Index_ i, Distance_ d, std::vector<Index_>* output_indices, std::vector<Distance_>* output_distances) {
        auto iptr = my_parent.my_data.data() + sanisizer::product_unsafe<std::size_t>(my_parent.my_new_locations[i], my_parent.my_dim);

        if (!output_indices && !output_distances) {
            Index_ count = 0;
            my_parent.template search_all<true>(0, iptr, d, count);
            return count_all_neighbors_without_self(count);

        } else {
            my_all_neighbors.clear();
            my_parent.template search_all<false>(0, iptr, d, my_all_neighbors);
            report_all_neighbors(my_all_neighbors, output_indices, output_distances, i);
            return count_all_neighbors_without_self(my_all_neighbors.size());
        }
    }

    Index_ search_all(const Data_* query, Distance_ d, std::vector<Index_>* output_indices, std::vector<Distance_>* output_distances) {
        if (my_parent.my_nodes.empty()) { // protect the search_all() method when there is not even a node 0 to start the recursion.
            my_all_neighbors.clear();
            report_all_neighbors(my_all_neighbors, output_indices, output_distances);
            return 0;
        }

        if (!output_indices && !output_distances) {
            Index_ count = 0;
            my_parent.template search_all<true>(0, query, d, count);
            return count;

        } else {
            my_all_neighbors.clear();
            my_parent.template search_all<false>(0, query, d, my_all_neighbors);
            report_all_neighbors(my_all_neighbors, output_indices, output_distances);
            return my_all_neighbors.size();
        }
    }
};

template<typename Index_, typename Data_, typename Distance_, class DistanceMetric_>
class VptreePrebuilt final : public Prebuilt<Index_, Data_, Distance_> {
private:
    std::size_t my_dim;
    Index_ my_obs;
    std::vector<Data_> my_data;
    std::shared_ptr<const DistanceMetric_> my_metric;

public:
    Index_ num_observations() const {
        return my_obs;
    } 

    std::size_t num_dimensions() const {
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
    Index_ build(Index_ lower, Index_ upper, const Data_* coords, std::vector<DataPoint>& items, Rng_& rng) {
        /* 
         * We're assuming that lower < upper at each point within this
         * recursion. This requires some protection at the call site
         * when nobs = 0, see the constructor.
         */

        Index_ pos = my_nodes.size();
        my_nodes.emplace_back();
        Node& node = my_nodes.back(); // this is safe during recursion because we reserved 'nodes' already to the number of observations, see the constructor.

        Index_ gap = upper - lower;
        if (gap > 1) { // not yet at a leaf.

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
            const Data_* vantage_ptr = coords + sanisizer::product_unsafe<std::size_t>(vantage.second, my_dim);

            // Compute distances to the new vantage point.
            for (Index_ i = lower + 1; i < upper; ++i) {
                const Data_* loc = coords + sanisizer::product_unsafe<std::size_t>(items[i].second, my_dim);
                items[i].first = my_metric->raw(my_dim, vantage_ptr, loc);
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
    VptreePrebuilt(std::size_t num_dim, Index_ num_obs, std::vector<Data_> data, std::shared_ptr<const DistanceMetric_> metric) : 
        my_dim(num_dim),
        my_obs(num_obs),
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
            const std::mt19937_64::result_type base = 1234567890, m1 = my_obs, m2 = my_dim;
            std::mt19937_64 rand(base * m1 + m2);

            build(0, my_obs, my_data.data(), items, rand);

            // Resorting data in place to match order of occurrence within
            // 'nodes', for better cache locality.
            auto used = sanisizer::create<std::vector<char> >(sanisizer::attest_gez(my_obs));
            auto buffer = sanisizer::create<std::vector<Data_> >(sanisizer::attest_gez(my_dim));
            sanisizer::resize(my_new_locations, sanisizer::attest_gez(my_obs));
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

                auto optr = host + sanisizer::product_unsafe<std::size_t>(o, my_dim);
                std::copy_n(optr, my_dim, buffer.begin());
                Index_ replacement = current.index;

                do {
                    auto rptr = host + sanisizer::product_unsafe<std::size_t>(replacement, my_dim);
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
    void search_nn(Index_ curnode_index, const Data_* target, Distance_& max_dist, NeighborQueue<Index_, Distance_>& nearest) const { 
        auto nptr = my_data.data() + sanisizer::product_unsafe<std::size_t>(curnode_index, my_dim);
        Distance_ dist = my_metric->normalize(my_metric->raw(my_dim, nptr, target));

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
        auto nptr = my_data.data() + sanisizer::product_unsafe<std::size_t>(curnode_index, my_dim);
        Distance_ dist = my_metric->normalize(my_metric->raw(my_dim, nptr, target));

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

    friend class VptreeSearcher<Index_, Data_, Distance_, DistanceMetric_>;

public:
    std::unique_ptr<Searcher<Index_, Data_, Distance_> > initialize() const {
        return initialize_known();
    }

    auto initialize_known() const {
        return std::make_unique<VptreeSearcher<Index_, Data_, Distance_, DistanceMetric_> >(*this);
    }

public:
    void save(const std::filesystem::path& dir) const {
        quick_save(dir / "ALGORITHM", vptree_prebuilt_save_name, std::strlen(vptree_prebuilt_save_name));
        quick_save(dir / "DATA", my_data.data(), my_data.size());
        quick_save(dir / "NUM_OBS", &my_obs, 1);
        quick_save(dir / "NUM_DIM", &my_dim, 1);
        quick_save(dir / "NODES", my_nodes.data(), my_nodes.size());
        quick_save(dir / "NEW_LOCATIONS", my_new_locations.data(), my_new_locations.size());

        const auto distdir = dir / "DISTANCE";
        std::filesystem::create_directory(distdir);
        my_metric->save(distdir);
    }

    VptreePrebuilt(const std::filesystem::path& dir) {
        quick_load(dir / "NUM_OBS", &my_obs, 1);
        quick_load(dir / "NUM_DIM", &my_dim, 1);

        my_data.resize(sanisizer::product<I<decltype(my_data.size())> >(sanisizer::attest_gez(my_obs), my_dim));
        quick_load(dir / "DATA", my_data.data(), my_data.size());

        sanisizer::resize(my_nodes, sanisizer::attest_gez(my_obs));
        quick_load(dir / "NODES", my_nodes.data(), my_nodes.size());

        sanisizer::resize(my_new_locations, sanisizer::attest_gez(my_obs));
        quick_load(dir / "NEW_LOCATIONS", my_new_locations.data(), my_new_locations.size());

        auto dptr = load_distance_metric_raw<Data_, Distance_>(dir / "DISTANCE");
        auto xptr = dynamic_cast<DistanceMetric_*>(dptr);
        if (xptr == NULL) {
            throw std::runtime_error("cannot cast the loaded distance metric to a DistanceMetric_");
        }
        my_metric.reset(xptr);
    }
};
/**
 * @endcond
 */

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
 * Note that the VP tree search is theoretically "exact" but the behavior of the implementation will be affected by round-off error for floating-point inputs.
 * Search decisions based on the triangle inequality may not be correct in some edge cases involving tied distances.
 * This manifests as a different selection of neighbors compared to an exhaustive search (e.g., by `BruteforceBuilder`),
 * typically when (i) an observation is equidistant to multiple other observations that are not duplicates of each other
 * and (ii) the tied distances occur at the `k`-th nearest neighbor for `Searcher::search()` or are tied with `threshold` for `Searcher::search_all()`.
 * In practice, any errors are very rare and can probably be ignored for most applications.
 * If more accuracy is required, a partial mitigation is to just increase `k` or `threshold` to reduce the risk of incorrect search decisions,
 * and then filter the results to the desired set of neighbors.
 *
 * @tparam Index_ Integer type for the observation indices.
 * @tparam Data_ Numeric type for the input and query data.
 * @tparam Distance_ Numeric type for the distances, usually floating-point.
 * @tparam Matrix_ Class of the input data matrix. 
 * This should satisfy the `Matrix` interface.
 * @tparam DistanceMetric_ Class implementing the distance metric calculation.
 * This should satisfy the `DistanceMetric` interface.
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
    typename Index_,
    typename Data_,
    typename Distance_,
    class Matrix_ = Matrix<Index_, Data_>,
    class DistanceMetric_ = DistanceMetric<Data_, Distance_>
>
class VptreeBuilder final : public Builder<Index_, Data_, Distance_, Matrix_> {
public:
    /**
     * @param metric Pointer to a distance metric instance, e.g., `EuclideanDistance`.
     */
    VptreeBuilder(std::shared_ptr<const DistanceMetric_> metric) : my_metric(std::move(metric)) {}

private:
    std::shared_ptr<const DistanceMetric_> my_metric;

public:
    Prebuilt<Index_, Data_, Distance_>* build_raw(const Matrix_& data) const {
        return build_known_raw(data);
    }

public:
    /**
     * Override to assist devirtualization.
     */
    auto build_known_raw(const Matrix_& data) const {
        std::size_t ndim = data.num_dimensions();
        Index_ nobs = data.num_observations();
        auto work = data.new_known_extractor();

        // We assume that that vector::size_type <= size_t, otherwise data() wouldn't be a contiguous array.
        std::vector<Data_> store(sanisizer::product<typename std::vector<Data_>::size_type>(ndim, sanisizer::attest_gez(nobs)));
        for (Index_ o = 0; o < nobs; ++o) {
            std::copy_n(work->next(), ndim, store.data() + sanisizer::product_unsafe<std::size_t>(o, ndim));
        }

        return new VptreePrebuilt<Index_, Data_, Distance_, DistanceMetric_>(ndim, nobs, std::move(store), my_metric);
    }

    /**
     * Override to assist devirtualization.
     */
    auto build_known_unique(const Matrix_& data) const {
        return std::unique_ptr<I<decltype(*build_known_raw(data))> >(build_known_raw(data));
    }

    /**
     * Override to assist devirtualization.
     */
    auto build_known_shared(const Matrix_& data) const {
        return std::shared_ptr<I<decltype(*build_known_raw(data))> >(build_known_raw(data));
    }
};

}

#endif
