#ifndef KNNCOLLE_SEARCHER_HPP
#define KNNCOLLE_SEARCHER_HPP

#include <vector>
#include <stdexcept>

/**
 * @file Searcher.hpp
 *
 * @brief Interface for searching nearest-neighbor indices.
 */

namespace knncolle {

/**
 * @brief Interface for searching nearest-neighbor search indices.
 *
 * Instances of `Searcher` subclasses are typically constructed with `Prebuilt::initialize()`.
 * This is intended to allow re-use of data allocations across different calls to `search()`.
 * Users should ensure that a `Searcher` instance does not outlive the `Prebuilt` object used to generate it;
 * this allows developers of the former to hold references to the latter.
 *
 * @tparam Index_ Integer type for the observation indices.
 * @tparam Data_ Numeric type for the query data.
 * @tparam Distance_ Numeric type for the distances, usually floating-point.
 */
template<typename Index_, typename Data_, typename Distance_>
class Searcher {
public:
    /**
     * @cond
     */
    Searcher() = default;
    Searcher(Searcher&&) = default;
    Searcher(const Searcher&) = default;
    Searcher& operator=(Searcher&&) = default;
    Searcher& operator=(const Searcher&) = default;
    virtual ~Searcher() = default;
    /**
     * @endcond
     */

    /** 
     * Find the nearest neighbors of the `i`-th observation in the dataset.
     *
     * @param i The index of the observation of interest.
     * This should be non-negative and less than the total number of observations in `Prebuilt::num_observations()`.
     * @param k The number of neighbors to identify.
     * This should be a non-negative integer that is less than the total number of observations in `Prebuilt::num_observations()`.
     * (Except if `Prebuilt::num_observations() == 0`, in which case the only valid choice for `k` is also zero.)
     * Users can call `cap_k()` to easily cap `k` based on `Prebuilt::num_observations()`.
     * @param[out] output_indices Pointer to a vector, to be filled with the identities of the nearest neighbors in order of increasing distance.
     * On output, the length of the vector should be equal to `k`.
     * All entries should be unique and less than `Prebuilt::num_observations()`.
     * This vector is guaranteed to not contain `i` itself.
     * Optionally NULL, in which case no indices are returned.
     * @param[out] output_distances Pointer to a vector, to be filled with the distances of the nearest neighbors. 
     * This corresponds to the indices reported in `output_indices`.
     * Optionally NULL, in which case no distances are returned.
     */
    virtual void search(Index_ i, Index_ k, std::vector<Index_>* output_indices, std::vector<Distance_>* output_distances) = 0;

    /** 
     * Find the nearest neighbors of a new observation.
     *
     * @param query Pointer to an array of length equal to `Prebuilt::num_dimensions()`, containing the coordinates of the query point.
     * @param k The number of neighbors to identify.
     * This should be a non-negative integer that is no greater than the number of observations in `Prebuilt::num_observations()`.
     * Users can call `cap_k_query()` to easily cap `k` based on `Prebuilt::num_observations()`.
     * @param[out] output_indices Pointer to a vector, to be filled with the identities of the nearest neighbors in order of increasing distance.
     * On output, the length of the vector should be equal to `k`.
     * All entries should be unique and less than `Prebuilt::num_observations()`.
     * Optionally NULL, in which case no indices are returned.
     * @param[out] output_distances Pointer to a vector, to be filled with the distances of the nearest neighbors. 
     * This corresponds to the indices reported in `output_indices`.
     * Optionally NULL, in which case no distances are returned.
     */
    virtual void search(const Data_* query, Index_ k, std::vector<Index_>* output_indices, std::vector<Distance_>* output_distances) = 0;

public:
    /**
     * It is expected that the return value of this method should be constant throughout the lifetime of any `Searcher` instance. 
     *
     * @return Whether this instance has an implementation of the `search_all()` method.
     */
    virtual bool can_search_all() const {
        return false;
    }

    /** 
     * Find all neighbors of the `i`-th observation within a certain distance.
     * This method is optional and may not be implemented by subclasses, so applications should check `can_search_all()` before attempting a call.
     * A run-time exception is thrown if this method is called on a subclass that does not implement it.
     *
     * @param i The index of the observation of interest.
     * This should be non-negative and less than the total number of observations in `Prebuilt::num_observations()`.
     * @param distance The distance in which to consider neighbors.
     * @param[out] output_indices Pointer to a vector, to be filled with the identities of the nearest neighbors in order of increasing distance.
     * This vector is guaranteed to not contain `i` itself.
     * Optionally NULL, in which case no indices are returned.
     * @param[out] output_distances Pointer to a vector, to be filled with the distances of the nearest neighbors. 
     * This corresponds to the indices reported in `output_indices`.
     * Optionally NULL, in which case no distances are returned.
     *
     * @return Number of neighbors within `distance` of `i`.
     */
    virtual Index_ search_all([[maybe_unused]] Index_ i, [[maybe_unused]] Distance_ distance, [[maybe_unused]] std::vector<Index_>* output_indices, [[maybe_unused]] std::vector<Distance_>* output_distances) {
        throw std::runtime_error("distance-based searches not supported");
        return 0;
    }

    /** 
     * Find all neighbors of a new observation within a certain distance.
     * This method is optional and may not be implemented by subclasses, so applications should check `can_search_all()` before attempting a call.
     * A run-time exception is thrown if this method is called on a subclass that does not implement it.
     *
     * @param query Pointer to an array of length equal to `Prebuilt::num_dimensions()`, containing the coordinates of the query point.
     * @param distance The distance in which to consider neighbors.
     * @param[out] output_indices Pointer to a vector, to be filled with the identities of the nearest neighbors in order of increasing distance.
     * Optionally NULL, in which case no indices are returned.
     * @param[out] output_distances Pointer to a vector, to be filled with the distances of the nearest neighbors. 
     * This corresponds to the indices reported in `output_indices`.
     * Optionally NULL, in which case no distances are returned.
     *
     * @return Number of neighbors within `distance` of `query`.
     */
    virtual Index_ search_all([[maybe_unused]] const Data_* query, [[maybe_unused]] Distance_ distance, [[maybe_unused]] std::vector<Index_>* output_indices, [[maybe_unused]] std::vector<Distance_>* output_distances) {
        throw std::runtime_error("distance-based searches not supported");
        return 0;
    }
};

}

#endif
