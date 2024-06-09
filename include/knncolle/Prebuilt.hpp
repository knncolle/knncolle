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
 * @tparam Index_ Integer type for the indices.
 * @tparam Float_ Floating point type for the query data and output distances.
 */
template<typename Dim_, typename Index_, typename Query_>
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
     * Get the vector of coordinates for a given observation in the dataset. 
     *
     * `buffer` may not be filled if a pointer to the internal data store can be returned directly.
     * This can be assumed to be the case if the return address is not the same as `buffer`.
     *
     * @param i Index of the observation.
     * This should be non-negative and less than the total number of observations in `num_observations()`.
     * @param[out] buffer Buffer of length `num_dimensions()` to store the coordinates.
     *
     * @return A pointer to an array containing the coordinate vector.
     *
     */
    virtual const Float_* observation(Index_ i, Float_* buffer) const = 0;

public:
    /** 
     * Find the nearest neighbors of the `i`-th observation in the dataset.
     *
     * @param i The index of the observation of interest.
     * This should be non-negative and less than the total number of observations in `num_observations()`.
     * @param k The number of neighbors to identify.
     *
     * @return A vector of (index, distance) pairs containing the identities of the nearest neighbors in order of increasing distance.
     * Length is at most `k` but may be shorter if the total number of observations is less than `k + 1`.
     * This vector is guaranteed to not contain `i` itself.
     */
    virtual std::vector<std::pair<Index_, Float_> > find_nearest_neighbors(Index_ i, Index_ k) const = 0;

    /** 
     * Find the nearest neighbors of a new observation.
     *
     * @param query Pointer to an array of length equal to `num_dimensions()`, containing the coordinates of the query point.
     * @param k The number of neighbors to identify.
     *
     * @return A vector of (index, distance) pairs containing the identities of the nearest neighbors in order of increasing distance.
     * Length is at most `k` but may be shorter if the total number of observations is less than `k`.
     */
    virtual std::vector<std::pair<Index_, Float_> > find_nearest_neighbors(const Float_* query, Index_ k) const = 0;
};

}

#endif
