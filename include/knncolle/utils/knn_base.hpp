#ifndef KNNCOLLE_BASE_HPP
#define KNNCOLLE_BASE_HPP

#include <vector>

/**
 * @file knn_base.hpp
 *
 * Defines the virtual base class for all methods in the **knncolle** library.
 */

namespace knncolle {

/**
 * @brief Virtual base class defining the **knncolle** interface.
 *
 * Defines the minimum set up methods supported by all subclasses implementing specific methods.
 *
 * @tparam INDEX_t Integer type for the indices.
 * @tparam DISTANCE_t Floating point type for the distances.
 * @tparam QUERY_t Floating point type for the query data.
 */
template<typename INDEX_t = int, typename DISTANCE_t = double, typename QUERY_t = double>
class knn_base {
public:
    /**
     * Get the number of observations in the dataset to be searched.
     */
    virtual INDEX_t nobs() const = 0;
    
    /**
     * Get the number of dimensions.
     */
    virtual INDEX_t ndim() const = 0;

    virtual ~knn_base() {}

public:
    /** 
     * Find the nearest neighbors of the `index`-th observation in the dataset.
     *
     * @param index The index of the observation of interest.
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
