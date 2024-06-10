#ifndef KNNCOLLE_PREBUILT_HPP
#define KNNCOLLE_PREBUILT_HPP

#include <vector>

/**
 * @file Prebuilt.hpp
 *
 * @brief Interface for prebuilt nearest-neighbor indices.
 */

namespace knncolle {

/**
 * @brief Interface for prebuilt nearest-neighbor search indices.
 *
 * @tparam Dim_ Integer type for the number of dimensions.
 * For the output of `Builder::build`, this is set to `MockMatrix::dimension_type`.
 * @tparam Index_ Integer type for the indices.
 * For the output of `Builder::build`, this is set to `MockMatrix::index_type`.
 * @tparam Float_ Floating point type for the query data and output distances.
 */
template<typename Dim_ = int, typename Index_ = int, typename Float_ = double>
class Prebuilt {
public:
    /**
     * @return Number of observations in the dataset to be searched.
     */
    virtual Index_ num_observations() const = 0;
    
    /**
     * @return Number of dimensions.
     */
    virtual Dim_ num_dimensions() const = 0;

    /**
     * @cond
     */
    virtual ~Prebuilt() = default;
    /**
     * @endcond
     */

public:
    /** 
     * Find the nearest neighbors of the `i`-th observation in the dataset.
     *
     * @param i The index of the observation of interest.
     * This should be non-negative and less than the total number of observations in `num_observations()`.
     * @param k The number of neighbors to identify.
     * @param[out] output On output, a vector of (index, distance) pairs containing the identities of the nearest neighbors in order of increasing distance.
     * Length is at most `k` but may be shorter if the total number of observations is less than `k + 1`.
     * This vector is guaranteed to not contain `i` itself.
     */
    virtual void search(Index_ i, Index_ k, std::vector<std::pair<Index_, Float_> >& output) const = 0;

    /** 
     * Find the nearest neighbors of a new observation.
     *
     * @param query Pointer to an array of length equal to `num_dimensions()`, containing the coordinates of the query point.
     * @param k The number of neighbors to identify.
     * @param[out] output On output, a vector of (index, distance) pairs containing the identities of the nearest neighbors in order of increasing distance.
     * Length is at most `k` but may be shorter if the total number of observations is less than `k`.
     */
    virtual void search(const Float_* query, Index_ k, std::vector<std::pair<Index_, Float_> >& output) const = 0;
};

}

#endif
