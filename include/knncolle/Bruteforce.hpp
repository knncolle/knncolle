#ifndef KNNCOLLE_BRUTEFORCE_HPP
#define KNNCOLLE_BRUTEFORCE_HPP

#include "distances.hpp"
#include "NeighborQueue.hpp"
#include "Builder.hpp"
#include "Prebuilt.hpp"
#include "MockMatrix.hpp"

#include <vector>
#include <type_traits>

/**
 * @file Bruteforce.hpp
 *
 * @brief Implements a brute-force search for nearest neighbors.
 */

namespace knncolle {

/**
 * @brief Index for a brute-force nearest neighbor search.
 *
 * Instances of this class are usually constructed using `BruteforceBuilder`.
 *
 * @tparam Distance_ A distance calculation class satisfying the `MockDistance` contract.
 * This may be set to a lower-precision type than `Float_` to save memory.
 * @tparam Dim_ Integer type for the number of dimensions.
 * For the output of `BruteforceBuilder::build`, this is set to `MockMatrix::dimension_type`.
 * @tparam Index_ Integer type for the indices.
 * For the output of `BruteforceBuilder::build`, this is set to `MockMatrix::index_type`.
 * @tparam Store_ Floating point type for the stored data. 
 * For the output of `BruteforceBuilder::build`, this is set to `MockMatrix::data_type`.
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
     * @param num_dim Number of dimensions.
     * @param num_obs Number of observations.
     * @param data Vector of length equal to `num_dim * num_obs`, containing a column-major matrix where rows are dimensions and columns are observations.
     */
    BruteforcePrebuilt(Dim_ num_dim, Index_ num_obs, std::vector<Store_> data) : 
        my_dim(num_dim), my_obs(num_obs), my_long_ndim(num_dim), my_data(std::move(data)) {}

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
        for (Index_ x = 0; x < my_obs; ++x, copy += my_dim) {
            nearest.add(x, Distance_::template raw_distance<Float_>(query, copy, my_dim));
        }
    }

    static void normalize(std::vector<std::pair<Index_, Float_> >& results) {
        for (auto& d : results) {
            d.second = Distance_::normalize(d.second);
        }
        return;
    } 

public:
    void search(Index_ i, Index_ k, std::vector<std::pair<Index_, Float_> >& output) const {
        internal::NeighborQueue<Index_, Float_> nearest(k + 1);
        auto ptr = my_data.data() + static_cast<size_t>(i) * my_long_ndim; // cast to avoid overflow.
        search(ptr, nearest);
        nearest.report(output, i);
        normalize(output);
    }

    void search(const Float_* query, Index_ k, std::vector<std::pair<Index_, Float_> >& output) const {
        internal::NeighborQueue<Index_, Float_> nearest(k);
        search(query, nearest);
        nearest.report(output);
        normalize(output);
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
