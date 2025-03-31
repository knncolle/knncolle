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

/**
 * @file Bruteforce.hpp
 *
 * @brief Implements a brute-force search for nearest neighbors.
 */

namespace knncolle {

template<class DistanceMetric_, typename Dim_, typename Index_, typename Data_, typename Distance_, typename Store_>
class BruteforcePrebuilt;

/**
 * @brief Brute-force nearest neighbor searcher.
 *
 * Instances of this class are usually constructed using `BruteforcePrebuilt::initialize()`.
 *
 * @tparam Dim_ Integer type for the number of dimensions.
 * @tparam Index_ Integer type for the indices.
 * @tparam Data_ Numeric type for the input and query data.
 * @tparam Distance_ Floating point type for the distances.
 * @tparam DistanceMetric_ Class that satisfies the `DistanceMetric_` interface.
 * @tparam Store_ Numeric type for the stored data.
 * This may be a lower-precision type than `Data_` to reduce memory usage.
 */
template<typename Dim_, typename Index_, typename Data_, typename Distance_, class DistanceMetric_, typename Store_>
class BruteforceSearcher final : public Searcher<Index_, Data_, Distance_> {
public:
    /**
     * @cond
     */
    BruteforceSearcher(const BruteforcePrebuilt<Dim_, Index_, Data_, Distance_, DistanceMetric_, Store_>& parent) : my_parent(parent) {}
    /**
     * @endcond
     */

private:                
    const BruteforcePrebuilt<Dim_, Index_, Data_, Distance_, DistanceMetric_, Store_>& my_parent;
    internal::NeighborQueue<Index_, Distance_> my_nearest;
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
        auto ptr = my_parent.my_data.data() + static_cast<size_t>(i) * my_parent.my_long_ndim; // cast to avoid overflow.
        my_parent.search(ptr, my_nearest);
        my_nearest.report(output_indices, output_distances, i);
        normalize(output_distances);
    }

    void search(const Data_* query, Index_ k, std::vector<Index_>* output_indices, std::vector<Distance_>* output_distances) {
        if (k == 0) { // protect the NeighborQueue from k = 0.
            internal::flush_output(output_indices, output_distances, 0);
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
        auto ptr = my_parent.my_data.data() + static_cast<size_t>(i) * my_parent.my_long_ndim; // cast to avoid overflow.

        if (!output_indices && !output_distances) {
            Index_ count = 0;
            my_parent.template search_all<true>(ptr, d, count);
            return internal::safe_remove_self(count);

        } else {
            my_all_neighbors.clear();
            my_parent.template search_all<false>(ptr, d, my_all_neighbors);
            internal::report_all_neighbors(my_all_neighbors, output_indices, output_distances, i);
            normalize(output_distances);
            return internal::safe_remove_self(my_all_neighbors.size());
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
            internal::report_all_neighbors(my_all_neighbors, output_indices, output_distances);
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
 * @tparam Dim_ Integer type for the number of dimensions.
 * @tparam Index_ Integer type for the indices.
 * @tparam Data_ Numeric type for the input and query data.
 * @tparam Distance_ Floating point type for the distances.
 * @tparam DistanceMetric_ Class that satisfies the `DistanceMetric_` interface.
 * @tparam Store_ Numeric type for the stored data.
 * This may be a lower-precision type than `Data_` to reduce memory usage.
 */
template<typename Dim_, typename Index_, typename Data_, typename Distance_, class DistanceMetric_, typename Store_>
class BruteforcePrebuilt final : public Prebuilt<Dim_, Index_, Data_, Distance_> {
private:
    Dim_ my_dim;
    Index_ my_obs;
    size_t my_long_ndim;
    std::vector<Store_> my_data;
    std::shared_ptr<const DistanceMetric_> my_metric;

public:
    /**
     * @cond
     */
    BruteforcePrebuilt(Dim_ num_dim, Index_ num_obs, std::vector<Store_> data, std::shared_ptr<const DistanceMetric_> metric) : 
        my_dim(num_dim), my_obs(num_obs), my_long_ndim(num_dim), my_data(std::move(data)), my_metric(std::move(metric)) {}
    /**
     * @endcond
     */

public:
    Dim_ num_dimensions() const {
        return my_dim;
    }

    Index_ num_observations() const {
        return my_obs;
    }

private:
    void search(const Data_* query, internal::NeighborQueue<Index_, Distance_>& nearest) const {
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

    friend class BruteforceSearcher<Dim_, Index_, Data_, Distance_, DistanceMetric_, Store_>;

public:
    /**
     * Creates a `BruteforceSearcher` instance.
     */
    std::unique_ptr<Searcher<Index_, Data_, Distance_> > initialize() const {
        return std::make_unique<BruteforceSearcher<Dim_, Index_, Data_, Distance_, DistanceMetric_, Store_> >(*this);
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
 * @tparam Dim_ Integer type for the number of dimensions.
 * @tparam Index_ Integer type for the indices.
 * @tparam Data_ Numeric type for the input and query data.
 * @tparam Distance_ Floating point type for the distances.
 * @tparam DistanceMetric_ Class that satisfies the `DistanceMetric_` interface.
 * @tparam Store_ Numeric type for the stored data.
 * This may be a lower-precision type than `Data_` to reduce memory usage.
 * @tparam Matrix_ Class that satisfies the `Matrix` interface.
 */
template<
    typename Dim_,
    typename Index_,
    typename Data_,
    typename Distance_,
    class DistanceMetric_ = DistanceMetric<Dim_, Data_, Distance_>,
    typename Store_ = Data_,
    class Matrix_ = Matrix<Dim_, Index_, Data_>
>
class BruteforceBuilder final : public Builder<Dim_, Index_, Data_, Distance_, Matrix_> {
public:
    /**
     * @param metric Pointer to a distance metric instance, e.g., `EuclideanDistance`.
     */
    BruteforceBuilder(std::shared_ptr<const DistanceMetric_> metric) : my_metric(std::move(metric)) {}

    /**
     * @param metric Pointer to a distance metric instance, e.g., `EuclideanDistance`.
     */
    BruteforceBuilder(const DistanceMetric_* metric) : BruteforceBuilder(std::shared_ptr<const DistanceMetric_>(metric)) {}

private:
    std::shared_ptr<const DistanceMetric_> my_metric;

public:
    /**
     * Creates a `BruteforcePrebuilt` instance.
     */
    Prebuilt<Dim_, Index_, Data_, Distance_>* build_raw(const Matrix_& data) const {
        size_t ndim = data.num_dimensions();
        size_t nobs = data.num_observations();
        auto work = data.new_extractor();

        std::vector<Store_> store(ndim * nobs);
        for (size_t o = 0; o < nobs; ++o) {
            std::copy_n(work->next(), ndim, store.begin() + o * ndim);
        }

        return new BruteforcePrebuilt<Dim_, Index_, Data_, Distance_, DistanceMetric_, Store_>(ndim, nobs, std::move(store), my_metric);
    }
};

}

#endif
