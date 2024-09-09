#ifndef KNNCOLLE_BRUTEFORCE_HPP
#define KNNCOLLE_BRUTEFORCE_HPP

#include "distances.hpp"
#include "NeighborQueue.hpp"
#include "Searcher.hpp"
#include "Builder.hpp"
#include "Prebuilt.hpp"
#include "MockMatrix.hpp"
#include "report_all_neighbors.hpp"

#include <vector>
#include <type_traits>
#include <limits>

/**
 * @file Bruteforce.hpp
 *
 * @brief Implements a brute-force search for nearest neighbors.
 */

namespace knncolle {

template<class Distance_, typename Dim_, typename Index_, typename Store_, typename Float_>
class BruteforcePrebuilt;

/**
 * @brief Brute-force nearest neighbor searcher.
 *
 * Instances of this class are usually constructed using `BruteforcePrebuilt::initialize()`.
 *
 * @tparam Distance_ A distance calculation class satisfying the `MockDistance` contract.
 * @tparam Dim_ Integer type for the number of dimensions.
 * @tparam Index_ Integer type for the indices.
 * @tparam Store_ Floating point type for the stored data. 
 * @tparam Float_ Floating point type for the query data and output distances.
 */
template<class Distance_, typename Dim_, typename Index_, typename Store_, typename Float_>
class BruteforceSearcher : public Searcher<Index_, Float_> {
public:
    /**
     * @cond
     */
    BruteforceSearcher(const BruteforcePrebuilt<Distance_, Dim_, Index_, Store_, Float_>* parent) : my_parent(parent) {}
    /**
     * @endcond
     */

private:                
    const BruteforcePrebuilt<Distance_, Dim_, Index_, Store_, Float_>* my_parent;
    internal::NeighborQueue<Index_, Float_> my_nearest;
    std::vector<std::pair<Float_, Index_> > my_all_neighbors;

private:
    static void normalize(std::vector<Float_>* output_distances) {
        if (output_distances) {
            for (auto& d : *output_distances) {
                d = Distance_::normalize(d);
            }
        } 
    }

public:
    void search(Index_ i, Index_ k, std::vector<Index_>* output_indices, std::vector<Float_>* output_distances) {
        my_nearest.reset(k + 1);
        auto ptr = my_parent->my_data.data() + static_cast<size_t>(i) * my_parent->my_long_ndim; // cast to avoid overflow.
        my_parent->search(ptr, my_nearest);
        my_nearest.report(output_indices, output_distances, i);
        normalize(output_distances);
    }

    void search(const Float_* query, Index_ k, std::vector<Index_>* output_indices, std::vector<Float_>* output_distances) {
        if (k == 0) { // protect the NeighborQueue from k = 0.
            internal::flush_output(output_indices, output_distances, 0);
        } else {
            my_nearest.reset(k);
            my_parent->search(query, my_nearest);
            my_nearest.report(output_indices, output_distances);
            normalize(output_distances);
        }
    }

    bool can_search_all() const {
        return true;
    }

    Index_ search_all(Index_ i, Float_ d, std::vector<Index_>* output_indices, std::vector<Float_>* output_distances) {
        auto ptr = my_parent->my_data.data() + static_cast<size_t>(i) * my_parent->my_long_ndim; // cast to avoid overflow.

        if (!output_indices && !output_distances) {
            Index_ count = 0;
            my_parent->template search_all<true>(ptr, d, count);
            return internal::safe_remove_self(count);

        } else {
            my_all_neighbors.clear();
            my_parent->template search_all<false>(ptr, d, my_all_neighbors);
            internal::report_all_neighbors(my_all_neighbors, output_indices, output_distances, i);
            normalize(output_distances);
            return internal::safe_remove_self(my_all_neighbors.size());
        }
    }

    Index_ search_all(const Float_* query, Float_ d, std::vector<Index_>* output_indices, std::vector<Float_>* output_distances) {
        if (!output_indices && !output_distances) {
            Index_ count = 0;
            my_parent->template search_all<true>(query, d, count);
            return count;

        } else {
            my_all_neighbors.clear();
            my_parent->template search_all<false>(query, d, my_all_neighbors);
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
 * @tparam Distance_ A distance calculation class satisfying the `MockDistance` contract.
 * @tparam Dim_ Integer type for the number of dimensions.
 * For the output of `BruteforceBuilder::build_raw()`, this is set to `Matrix_::dimension_type`.
 * @tparam Index_ Integer type for the indices.
 * For the output of `BruteforceBuilder::build_raw()`, this is set to `Matrix_::index_type`.
 * @tparam Store_ Floating point type for the stored data. 
 * For the output of `BruteforceBuilder::build_raw()`, this is set to `Matrix_::data_type`.
 * This may be set to a lower-precision type than `Float_` to save memory.
 * @tparam Float_ Floating point type for the query data and output distances.
 */
template<class Distance_, typename Dim_, typename Index_, typename Store_, typename Float_>
class BruteforcePrebuilt : public Prebuilt<Dim_, Index_, Float_> {
private:
    Dim_ my_dim;
    Index_ my_obs;
    size_t my_long_ndim;
    std::vector<Store_> my_data;

public:
    /**
     * @cond
     */
    BruteforcePrebuilt(Dim_ num_dim, Index_ num_obs, std::vector<Store_> data) : 
        my_dim(num_dim), my_obs(num_obs), my_long_ndim(num_dim), my_data(std::move(data)) {}
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
    template<typename Query_>
    void search(const Query_* query, internal::NeighborQueue<Index_, Float_>& nearest) const {
        auto copy = my_data.data();
        Float_ threshold_raw = std::numeric_limits<Float_>::infinity();
        for (Index_ x = 0; x < my_obs; ++x, copy += my_dim) {
            auto dist_raw = Distance_::template raw_distance<Float_>(query, copy, my_dim);
            if (dist_raw <= threshold_raw) {
                nearest.add(x, dist_raw);
                if (nearest.is_full()) {
                    threshold_raw = nearest.limit();
                }
            }
        }
    }

    template<bool count_only_, typename Query_, typename Output_>
    void search_all(const Query_* query, Float_ threshold, Output_& all_neighbors) const {
        Float_ threshold_raw = Distance_::denormalize(threshold);
        auto copy = my_data.data();
        for (Index_ x = 0; x < my_obs; ++x, copy += my_dim) {
            Float_ raw_distance = Distance_::template raw_distance<Float_>(query, copy, my_dim);
            if (threshold_raw >= raw_distance) {
                if constexpr(count_only_) {
                    ++all_neighbors; // expect this to be an integer.
                } else {
                    all_neighbors.emplace_back(raw_distance, x); // expect this to be a vector of (distance, index) pairs.
                }
            }
        }
    }

    friend class BruteforceSearcher<Distance_, Dim_, Index_, Store_, Float_>;

public:
    /**
     * Creates a `BruteforceSearcher` instance.
     */
    std::unique_ptr<Searcher<Index_, Float_> > initialize() const {
        return std::make_unique<BruteforceSearcher<Distance_, Dim_, Index_, Store_, Float_> >(this);
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
 * @tparam Distance_ A distance calculation class satisfying the `MockDistance` contract.
 * @tparam Matrix_ Matrix-like type that satisfies the `MockMatrix` interface.
 * @tparam Float_ Floating point type for the query data and output distances.
 */
template<class Distance_ = EuclideanDistance, class Matrix_ = SimpleMatrix<int, int, double>, typename Float_ = double>
class BruteforceBuilder : public Builder<Matrix_, Float_> {
public:
    /**
     * Creates a `BruteforcePrebuilt` instance.
     */
    Prebuilt<typename Matrix_::dimension_type, typename Matrix_::index_type, Float_>* build_raw(const Matrix_& data) const {
        auto ndim = data.num_dimensions();
        auto nobs = data.num_observations();

        typedef decltype(ndim) Dim_;
        typedef decltype(nobs) Index_;
        typedef typename Matrix_::data_type Store_;
        std::vector<typename Matrix_::data_type> store(static_cast<size_t>(ndim) * static_cast<size_t>(nobs));

        auto work = data.create_workspace();
        auto sIt = store.begin();
        for (decltype(nobs) o = 0; o < nobs; ++o, sIt += ndim) {
            auto ptr = data.get_observation(work);
            std::copy(ptr, ptr + ndim, sIt);
        }

        return new BruteforcePrebuilt<Distance_, Dim_, Index_, Store_, Float_>(ndim, nobs, std::move(store));
    }
};

}

#endif
