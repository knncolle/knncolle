#ifndef KNNCOLLE_BRUTEFORCE_HPP
#define KNNCOLLE_BRUTEFORCE_HPP

#include "distances.hpp"
#include "NeighborQueue.hpp"
#include "Searcher.hpp"
#include "Builder.hpp"
#include "Prebuilt.hpp"
#include "Matrix.hpp"
#include "report_all_neighbors.hpp"

#include <vector>
#include <type_traits>
#include <limits>
#include <memory>
#include <cstddef>

/**
 * @file Bruteforce.hpp
 *
 * @brief Implements a brute-force search for nearest neighbors.
 */

namespace knncolle {

template<typename Index_, typename Data_, typename Distance_, typename DistanceMetric_>
class BruteforcePrebuilt;

/**
 * @brief Brute-force nearest neighbor searcher.
 *
 * Instances of this class are usually constructed using `BruteforcePrebuilt::initialize()`.
 *
 * @tparam Index_ Integer type for the indices.
 * @tparam Data_ Numeric type for the input and query data.
 * @tparam Distance_ Floating point type for the distances.
 * @tparam DistanceMetric_ Class implementing the distance metric calculation.
 * This should satisfy the `DistanceMetric` interface.
 */
template<typename Index_, typename Data_, typename Distance_, class DistanceMetric_>
class BruteforceSearcher final : public Searcher<Index_, Data_, Distance_> {
public:
    /**
     * @cond
     */
    BruteforceSearcher(const BruteforcePrebuilt<Index_, Data_, Distance_, DistanceMetric_>& parent) : my_parent(parent) {}
    /**
     * @endcond
     */

private:                
    const BruteforcePrebuilt<Index_, Data_, Distance_, DistanceMetric_>& my_parent;
    NeighborQueue<Index_, Distance_> my_nearest;
    std::vector<std::pair<Distance_, Index_> > my_all_neighbors;

private:
    void normalize(std::vector<Distance_>* output_distances) const {
        if (output_distances) {
            for (auto& d : *output_distances) {
                d = my_parent.my_metric->normalize(d);
            }
        } 
    }

public:
    void search(Index_ i, Index_ k, std::vector<Index_>* output_indices, std::vector<Distance_>* output_distances) {
        my_nearest.reset(k + 1);
        auto ptr = my_parent.my_data.data() + static_cast<std::size_t>(i) * my_parent.my_dim; // cast to avoid overflow.
        my_parent.search(ptr, my_nearest);
        my_nearest.report(output_indices, output_distances, i);
        normalize(output_distances);
    }

    void search(const Data_* query, Index_ k, std::vector<Index_>* output_indices, std::vector<Distance_>* output_distances) {
        if (k == 0) { // protect the NeighborQueue from k = 0.
            if (output_indices) {
                output_indices->clear();
            }
            if (output_distances) {
                output_distances->clear();
            }
        } else {
            my_nearest.reset(k);
            my_parent.search(query, my_nearest);
            my_nearest.report(output_indices, output_distances);
            normalize(output_distances);
        }
    }

    bool can_search_all() const {
        return true;
    }

    Index_ search_all(Index_ i, Distance_ d, std::vector<Index_>* output_indices, std::vector<Distance_>* output_distances) {
        auto ptr = my_parent.my_data.data() + static_cast<std::size_t>(i) * my_parent.my_dim; // cast to avoid overflow.

        if (!output_indices && !output_distances) {
            Index_ count = 0;
            my_parent.template search_all<true>(ptr, d, count);
            return count_all_neighbors_without_self(count);

        } else {
            my_all_neighbors.clear();
            my_parent.template search_all<false>(ptr, d, my_all_neighbors);
            report_all_neighbors(my_all_neighbors, output_indices, output_distances, i);
            normalize(output_distances);
            return count_all_neighbors_without_self(my_all_neighbors.size());
        }
    }

    Index_ search_all(const Data_* query, Distance_ d, std::vector<Index_>* output_indices, std::vector<Distance_>* output_distances) {
        if (!output_indices && !output_distances) {
            Index_ count = 0;
            my_parent.template search_all<true>(query, d, count);
            return count;

        } else {
            my_all_neighbors.clear();
            my_parent.template search_all<false>(query, d, my_all_neighbors);
            report_all_neighbors(my_all_neighbors, output_indices, output_distances);
            normalize(output_distances);
            return my_all_neighbors.size();
        }
    }
};

/**
 * @brief Index for a brute-force nearest neighbor search.
 *
 * Instances of this class are usually constructed using `BruteforceBuilder::build_raw()`.
 *
 * @tparam Index_ Integer type for the indices.
 * @tparam Data_ Numeric type for the input and query data.
 * @tparam Distance_ Floating point type for the distances.
 * @tparam DistanceMetric_ Class implementing the distance metric calculation.
 * This should satisfy the `DistanceMetric` interface.
 */
template<typename Index_, typename Data_, typename Distance_, class DistanceMetric_>
class BruteforcePrebuilt final : public Prebuilt<Index_, Data_, Distance_> {
private:
    std::size_t my_dim;
    Index_ my_obs;
    std::vector<Data_> my_data;
    std::shared_ptr<const DistanceMetric_> my_metric;

public:
    /**
     * @cond
     */
    BruteforcePrebuilt(std::size_t num_dim, Index_ num_obs, std::vector<Data_> data, std::shared_ptr<const DistanceMetric_> metric) : 
        my_dim(num_dim), my_obs(num_obs), my_data(std::move(data)), my_metric(std::move(metric)) {}
    /**
     * @endcond
     */

public:
    std::size_t num_dimensions() const {
        return my_dim;
    }

    Index_ num_observations() const {
        return my_obs;
    }

private:
    void search(const Data_* query, NeighborQueue<Index_, Distance_>& nearest) const {
        auto copy = my_data.data();
        Distance_ threshold_raw = std::numeric_limits<Distance_>::infinity();
        for (Index_ x = 0; x < my_obs; ++x, copy += my_dim) {
            auto dist_raw = my_metric->raw(my_dim, query, copy);
            if (dist_raw <= threshold_raw) {
                nearest.add(x, dist_raw);
                if (nearest.is_full()) {
                    threshold_raw = nearest.limit();
                }
            }
        }
    }

    template<bool count_only_, typename Output_>
    void search_all(const Data_* query, Distance_ threshold, Output_& all_neighbors) const {
        Distance_ threshold_raw = my_metric->denormalize(threshold);
        auto copy = my_data.data();
        for (Index_ x = 0; x < my_obs; ++x, copy += my_dim) {
            Distance_ raw = my_metric->raw(my_dim, query, copy);
            if (threshold_raw >= raw) {
                if constexpr(count_only_) {
                    ++all_neighbors; // expect this to be an integer.
                } else {
                    all_neighbors.emplace_back(raw, x); // expect this to be a vector of (distance, index) pairs.
                }
            }
        }
    }

    friend class BruteforceSearcher<Index_, Data_, Distance_, DistanceMetric_>;

public:
    /**
     * Creates a `BruteforceSearcher` instance.
     */
    std::unique_ptr<Searcher<Index_, Data_, Distance_> > initialize() const {
        return std::make_unique<BruteforceSearcher<Index_, Data_, Distance_, DistanceMetric_> >(*this);
    }
};

/**
 * @brief Perform a brute-force nearest neighbor search.
 *
 * The brute-force search computes all pairwise distances between data and query points to identify nearest neighbors of the latter.
 * It has quadratic complexity and is theoretically the worst-performing method;
 * however, it has effectively no overhead from constructing or querying indexing structures, 
 * potentially making it faster in cases where indexing provides little benefit (e.g., few data points, high dimensionality).
 *
 * @tparam Index_ Integer type for the indices.
 * @tparam Data_ Numeric type for the input and query data.
 * @tparam Distance_ Floating point type for the distances.
 * @tparam Matrix_ Class of the input data matrix. 
 * This should satisfy the `Matrix` interface.
 * @tparam DistanceMetric_ Class implementing the distance metric calculation.
 * This should satisfy the `DistanceMetric` interface.
 */
template<
    typename Index_,
    typename Data_,
    typename Distance_,
    class Matrix_ = Matrix<Index_, Data_>,
    class DistanceMetric_ = DistanceMetric<Data_, Distance_>
>
class BruteforceBuilder final : public Builder<Index_, Data_, Distance_, Matrix_> {
public:
    /**
     * @param metric Pointer to a distance metric instance, e.g., `EuclideanDistance`.
     */
    BruteforceBuilder(std::shared_ptr<const DistanceMetric_> metric) : my_metric(std::move(metric)) {}

private:
    std::shared_ptr<const DistanceMetric_> my_metric;

public:
    /**
     * Creates a `BruteforcePrebuilt` instance.
     */
    Prebuilt<Index_, Data_, Distance_>* build_raw(const Matrix_& data) const {
        std::size_t ndim = data.num_dimensions();
        Index_ nobs = data.num_observations();
        auto work = data.new_extractor();

        std::vector<Data_> store(ndim * static_cast<std::size_t>(nobs)); // cast to avoid overflow.
        for (Index_ o = 0; o < nobs; ++o) {
            std::copy_n(work->next(), ndim, store.begin() + static_cast<std::size_t>(o) * ndim); // cast to size_t to avoid overflow.
        }

        return new BruteforcePrebuilt<Index_, Data_, Distance_, DistanceMetric_>(ndim, nobs, std::move(store), my_metric);
    }
};

}

#endif
