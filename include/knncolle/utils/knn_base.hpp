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
     * @param[out] indices Vector to store the index of the nearest neighbors. 
     * Vector is resized to length no greater than `k` (but possibly less, if the total number of observations is less than `k`).
     * @param[out] distances Vector to store the distances to the nearest neighbors.
     * Each distance corresponds to a neighbor in `indices`. 
     * Guaranteed to be sorted in increasing order.
     * @param report_indices Whether to report indices.
     * @param report_distances Whether to report distances.
     * @param check_ties Whether to check for tied neighbors.
     *
     * @return 
     * If `report_indices = true`, `indices` is filled with the identities of the `k` nearest neighbors.
     * If `report_distances = true`, `distances` is filled with the distances to the `k` nearest neighbors.
     * If `check_ties = true`, a boolean is returned indicating whether ties were detected in the first `k + 1` neighbors, otherwise `false` is always returned.
     */
    virtual bool find_nearest_neighbors(ITYPE index, int k, std::vector<ITYPE>& indices, std::vector<DTYPE>& distances, 
        bool report_indices = true, bool report_distances = true, bool check_ties = true) const = 0;

    /** 
     * Find the nearest neighbors of a new observation.
     *
     * @param query Pointer to an array of length equal to `ndims()`, containing the coordinates of the query point.
     * @param k The number of neighbors to identify.
     * @param[out] indices Vector to store the index of the nearest neighbors. 
     * Vector is resized to length no greater than `k` (but possibly less, if the total number of observations is less than `k`).
     * @param[out] distances Vector to store the distances to the nearest neighbors.
     * Each distance corresponds to a neighbor in `indices`. 
     * Guaranteed to be sorted in increasing order.
     * @param report_indices Whether to report indices.
     * @param report_distances Whether to report distances.
     * @param check_ties Whether to check for tied neighbors.
     *
     * @return 
     * If `report_indices = true`, `indices` is filled with the identities of the `k` nearest neighbors.
     * If `report_distances = true`, `distances` is filled with the distances to the `k` nearest neighbors.
     * If `check_ties = true`, a boolean is returned indicating whether ties were detected in the first `k + 1` neighbors, otherwise `false` is always returned.
     */
    virtual bool find_nearest_neighbors(const DTYPE* query, int k, std::vector<ITYPE>& indices, std::vector<DTYPE>& distances,
        bool report_indices = true, bool report_distances = true, bool check_ties = true) const = 0;
};

}

#endif
