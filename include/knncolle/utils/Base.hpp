#ifndef KNNCOLLE_BASE_HPP
#define KNNCOLLE_BASE_HPP

#include <vector>

/**
 * @file Base.hpp
 *
 * @brief Defines the virtual base class for all **knncolle** methods.
 */

namespace knncolle {

/**
 * @brief Virtual base class defining the **knncolle** interface.
 *
 * Defines the minimum set of methods, to be implemented by all concrete subclasses. 
 *
 * @tparam INDEX_t Integer type for the indices.
 * @tparam DISTANCE_t Floating point type for the distances.
 * @tparam QUERY_t Floating point type for the query data.
 */
template<typename INDEX_t = int, typename DISTANCE_t = double, typename QUERY_t = DISTANCE_t>
class Base {
public:
    /**
     * Get the number of observations in the dataset to be searched.
     */
    virtual INDEX_t nobs() const = 0;
    
    /**
     * Get the number of dimensions.
     */
    virtual INDEX_t ndim() const = 0;

    virtual ~Base() {}

public:
    /**
     * Get the vector of coordinates for a given observation in the dataset. 
     * Type conversions may be performed if `QUERY_t` differs from the type of the internal data store.
     *
     * `buffer` may not be filled if a pointer to the internal data store can be returned directly.
     * This can be assumed to be the case if the return address is not the same as `buffer`.
     *
     * @param index Index of the observation.
     * This should be non-negative and less than the total number of observations in `nobs()`.
     * @param buffer Buffer to store the coordinates.
     *
     * @return A pointer to an array containing the coordinate vector.
     *
     */
    virtual const QUERY_t* observation(INDEX_t index, QUERY_t* buffer) const = 0;

    /**
     * Get the vector of coordinates for a given observation in the dataset. 
     * Type conversions may be performed if `QUERY_t` differs from the type of the internal data store.
     *
     * @param index Index of the observation.
     *
     * @return A vector of coordinates.
     *
     */
    virtual std::vector<QUERY_t> observation(INDEX_t index) const {
        std::vector<QUERY_t> output(ndim());
        auto ptr = observation(index, output.data());
        if (ptr != output.data()) {
            std::copy(ptr, ptr + output.size(), output.data());
        }
        return output;
    }

public:
    /** 
     * Find the nearest neighbors of the `index`-th observation in the dataset.
     *
     * @param index The index of the observation of interest.
     * This should be non-negative and less than the total number of observations in `nobs()`.
     * @param k The number of neighbors to identify.
     *
     * @return A vector of (index, distance) pairs containing the identities of the nearest neighbors in order of increasing distance.
     * Length is at most `k` but may be shorter if the total number of observations is less than `k + 1`.
     */
    virtual std::vector<std::pair<INDEX_t, DISTANCE_t> > find_nearest_neighbors(INDEX_t index, int k) const = 0;

    /** 
     * Find the nearest neighbors of a new observation.
     *
     * @param query Pointer to an array of length equal to `ndims()`, containing the coordinates of the query point.
     * @param k The number of neighbors to identify.
     *
     * @return A vector of (index, distance) pairs containing the identities of the nearest neighbors in order of increasing distance.
     * Length is at most `k` but may be shorter if the total number of observations is less than `k`.
     */
    virtual std::vector<std::pair<INDEX_t, DISTANCE_t> > find_nearest_neighbors(const QUERY_t* query, int k) const = 0;
};

}

#endif
