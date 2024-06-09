#ifndef KNNCOLLE_BRUTEFORCE_HPP
#define KNNCOLLE_BRUTEFORCE_HPP

#include "distances.hpp"
#include "NeighborQueue.hpp"
#include "Builder.hpp"
#include "Prebuilt.hpp"

#include <vector>
#include <type_traits>

/**
 * @file BruteForce.hpp
 *
 * @brief Implements a brute-force search for nearest neighbors.
 */

namespace knncolle {

/**
 * @brief Index for a brute-force nearest neighbor search.
 *
 * Instances of this class are usually constructed using `BruteForceBuilder`.
 *
 * @tparam Distance_ A distance calculation class satisfying the `MockDistance` contract.
 * @tparam Store_ Floating point type for the stored data. 
 * @tparam Dim_ Integer type for the number of dimensions.
 * @tparam Index_ Integer type for the indices.
 * @tparam Float_ Floating point type for the query data and output distances.
 */
template<class Distance_, typename Store_, typename Dim_, typename Index_, typename Float_>
class BruteForcePrebuilt : public Prebuilt<Dim_, Index_, Float_> {
private:
    Dim_ my_dim;
    Index_ my_obs;
    std::vector<Store_> my_data;

public:
    /**
     * @param num_dim Number of dimensions.
     * @param num_obs Number of observations.
     * @param data Vector of length equal to `num_dim * num_obs`, containing a column-major matrix where rows are dimensions and columns are observations.
     */
    BruteForcePrebuilt(Dim_ num_dim, Index_ num_obs, std::vector<Store_> data) : my_dim(num_dim), my_obs(num_obs), my_data(std::move(data)) {}

public:
    Dim_ num_dimensions() const {
        return my_dim;
    }

    Index_ num_observations() const {
        return my_obs;
    }

private:
    static void normalize(std::vector<std::pair<Index_, Float_> >& results) const {
        for (auto& d : results) {
            d.second = Distance_::normalize(d.second);
        }
        return;
    } 

public:
    std::vector<std::pair<Index_, Float_> > find_nearest_neighbors(Index_ i, int k) const {
        NeighborQueue<Index_, Float_> nearest(k);

        auto copy = my_store.data();
        for (Index_ x = 0; x < i; ++x, copy += my_dim) {
            nearest.add(x, Distance_::template raw_distance<Float_>(query, copy, my_dim));
        }
        copy += my_dim; // skip 'i' itself.
        for (Index_ x = i + 1; x < my_obs; ++x, copy += my_dim) {
            nearest.add(x, Distance_::template raw_distance<Float_>(query, copy, my_dim));
        }

        auto results = nearest.report();
        normalize(results);
        return results;
    }

    std::vector<std::pair<Index_, Float_> > find_nearest_neighbors(const Float_* query, int k) const {
        NeighborQueue<Index_, Float_> nearest(k);

        auto copy = my_store.data();
        for (Index_ x = 0; x < my_obs; ++x, copy += my_dim) {
            nearest.add(x, Distance_::template raw_distance<Float_>(query, copy, my_dim));
        }

        auto results = nearest.report();
        normalize(results);
        return results;
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
template<class Distance_ = EuclideanDistance, class MockMatrix_ = SimpleMatrix<double, int>, typename Float_ = double>
class BruteForceBuilder : public Builder<MockMatrix_, Float> {
public:
    /**
     * @param data A matrix-like object containing the data to be searched.
     * @return Pointer to a `BruteForcePrebuilt` object.
     */
    Prebuilt<typename Matrix_::dimension_type, typename Matrix_::index_type, Float_>* build(const MockMatrix_& data) const {
        auto ndim = data.num_dimensions();
        auto nobs = data.num_observations();

        typedef decltype(ndim) Dim_;
        typedef decltype(nobs) Index_;
        typedef typename Matrix_::data_type Store_;
        std::vector<typename Matrix::data_type> store(static_cast<size_t>(ndim) * static_cast<size_t>(nobs));

        auto work = data.create_workspace();
        auto sIt = store.begin();
        for (decltype(nobs) o = 0; o < nobs; ++o, sIt += ndim) {
            auto ptr = data.get_observation(obs);
            std::copy(ptr, ptr + ndim, sIt);
        }

        return new BruteForcePrebuilt<Distance_, Store_, Dim_, Index_, Float_>(ndim, nobs, std::move(store));
    }
};

}

#endif
