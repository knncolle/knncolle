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
 * @tparam ITYPE Integer type for the indices.
 * @tparam DTYPE Floating point type for the distances.
 */
template<typename ITYPE = int, typename DTYPE = double>
class knn_base {
public:
    /**
     * Get the number of observations in the dataset to be searched.
     */
    virtual ITYPE nobs() const = 0;
    
    /**
     * Get the number of dimensions.
     */
    virtual ITYPE ndim() const = 0;

    virtual ~knn_base() {}

public:
    /** 
     * Find the nearest neighbors of the `index`-th observation in the dataset.
     *
     * @param index The index of the observation of interest.
     * @param k The number of neighbors to identify.
     * @param[out] indices Pointer to a vector to store the index of the nearest neighbors. 
     * Vector is resized to length no greater than `k` (but possibly less, if the total number of observations is less than `k`).
     * If `NULL`, no indices are returned.
     * @param[out] distances Pointer to a vector to store the distances to the nearest neighbors.
     * Each distance corresponds to a neighbor in `indices`. 
     * Guaranteed to be sorted in increasing order.
     * If `NULL`, no distances are returned.
     *
     * @return 
     * If `report_indices = true`, `*indices` is filled with the identities of the `k` nearest neighbors.
     * If `report_distances = true`, `*distances` is filled with the distances to the `k` nearest neighbors.
     */
    virtual void find_nearest_neighbors(ITYPE index, int k, std::vector<ITYPE>* indices, std::vector<DTYPE>* distances) const = 0;

    /** 
     * Find the nearest neighbors of a new observation.
     *
     * @param query Pointer to an array of length equal to `ndims()`, containing the coordinates of the query point.
     * @param k The number of neighbors to identify.
     * @param[out] indices Pointer to a vector to store the index of the nearest neighbors. 
     * Vector is resized to length no greater than `k` (but possibly less, if the total number of observations is less than `k`).
     * If `NULL`, no indices are returned.
     * @param[out] distances Pointer to a vector to store the distances to the nearest neighbors.
     * Each distance corresponds to a neighbor in `indices`. 
     * Guaranteed to be sorted in increasing order.
     * If `NULL`, no distances are returned.
     *
     * @return 
     * If `report_indices = true`, `*indices` is filled with the identities of the `k` nearest neighbors.
     * If `report_distances = true`, `*distances` is filled with the distances to the `k` nearest neighbors.
     */
    virtual void find_nearest_neighbors(const DTYPE* query, int k, std::vector<ITYPE>* indices, std::vector<DTYPE>* distances) const = 0;
};

}

#endif
