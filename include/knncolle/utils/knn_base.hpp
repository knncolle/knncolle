#ifndef KNNCOLLE_BASE_HPP
#define KNNCOLLE_BASE_HPP

#include <vector>
#include "utils.hpp"

/**
 * @file base.hpp
 *
 * Defines the virtual base class for the **knncolle** library.
 */

namespace knncolle {

/**
 * @brief Virtual base class defining the **knncolle** interface.
 *
 * Defines the minimum set up methods supported by all subclasses implementing specific methods.
 */
class knn_base {
public:
    /**
     * Get the number of observations.
     */
    virtual CellIndex_t nobs() const = 0;
    
    /**
     * Get the number of dimensions.
     */
    virtual MatDim_t ndims() const = 0;

    virtual ~knn_base() {}
protected:
    std::vector<CellIndex_t> current_neighbors;

    std::vector<double> current_distances;

    bool current_tied;

public:
    /**
     * Get the indices of the nearest neighbors identified from the last `find_nearest_neighbors()` call.
     * This should only be used if `store_indices()` is set to `true`.
     *
     * Neighbors are guaranteed to be ordered by increasing distance from the query point.
     */
    const std::vector<CellIndex_t>& neighbors() const { return current_neighbors; }

    /**
     * Get the distances to the nearest neighbors identified from the last `find_nearest_neighbors()` call.
     * This should only be used if `store_distances()` is set to `true`.
     *
     * Distances are guaranteed to be in increasing order.
     */
    const std::vector<double>& distances() const { return current_distances; }

    /**
     * Get the tied status from the last `find_nearest_neighbors()` call.
     * This should only be used if `check_ties()` is set to `true`.
     */
    bool tied() const { return current_tied; }

protected:
    bool get_index = true;
    bool get_distance = true;
    bool get_ties = true;

public:
    /**
     * @param s Whether the indices of the nearest neighbors should be recorded in `find_nearest_neighbors()`.
     * Setting to `false` can improve efficiency if only the distances are of interest.
     */
    void store_indices(bool s = true) {
        get_index = s;
        return;
    }

    /**
     * @param s Whether the indices of the nearest neighbors should be recorded in `find_nearest_neighbors()`.
     * Setting to `false` can improve efficiency if only the indices are of interest.
     */
    void store_distances(bool s = true) {
        get_distance = s;
        return;
    }

    /**
     * @param w Whether to check for tied neighbors.
     * Setting to `false` can improve efficiency.
     */
    void check_ties(bool w = true) {
        get_ties = w;
        return;
    }

public:
    /** 
     * Find the nearest neighbors of the `index`-th observation in the dataset.
     *
     * @param index The index of the observation of interest.
     * @param k The number of neighbors to identify.
     *
     * @return The results of the search can be extracted with `neighbors()`, `distances()` and `tied()`.
     */
    virtual void find_nearest_neighbors(CellIndex_t index, NumNeighbors_t k) = 0;

    /** 
     * Find the nearest neighbors of a new observation.
     *
     * @param query Pointer to an array of length equal to `ndims()`, containing the coordinates of the query point.
     * @param k The number of neighbors to identify.
     *
     * @return The results of the search can be extracted with `neighbors()`, `distances()` and `tied()`.
     */
    virtual void find_nearest_neighbors(const double* query, NumNeighbors_t k) = 0;
};

}

#endif
