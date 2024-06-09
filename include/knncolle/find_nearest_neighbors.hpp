#ifndef KNNCOLLE_FIND_NEAREST_NEIGHBORS_HPP
#define KNNCOLLE_FIND_NEAREST_NEIGHBORS_HPP

#include <vector>
#include <utility>
#include <type_traits>
#include "Base.hpp"

/**
 * @file find_nearest_neighbors.hpp
 *
 * @brief Find nearest neighbors from an existing index.
 */

namespace knncolle {

/**
 * List of nearest neighbors for multiple observations.
 * Each entry corresponds to an observation and contains the nearest neighbors as (index, distance) pairs for that observation.
 *
 * @tparam INDEX_t Integer type for the indices.
 * @tparam DISTANCE_t Floating point type for the distances.
 */
template<typename INDEX_t = int, typename DISTANCE_t = double> 
using NeighborList = std::vector<std::vector<std::pair<INDEX_t, DISTANCE_t> > >;

/**
 * Find the nearest neighbors within a pre-built index.
 * This is a convenient wrapper around `Base::find_nearest_neighbors` that saves the caller the trouble of writing a loop.
 *
 * @tparam INDEX_t Integer type for the indices in the output object.
 * @tparam DISTANCE_t Floating point type for the distances in the output object
 * @tparam InputINDEX_t Integer type for the indices in the input index.
 * @tparam InputDISTANCE_t Floating point type for the distances in the input index.
 * @tparam QUERY_t Floating point type for the query data in the input index.
 *
 * @param ptr Pointer to a `Base` index.
 * @param k Number of nearest neighbors. 
 * @param nthreads Number of threads to use.
 *
 * @return A `NeighborList` of length equal to the number of observations in `ptr->nobs()`.
 * Each entry contains the `k` nearest neighbors for each observation, sorted by increasing distance.
 */
template<typename INDEX_t = int, typename DISTANCE_t = double, typename InputINDEX_t, typename InputDISTANCE_t, typename InputQUERY_t> 
NeighborList<INDEX_t, DISTANCE_t> find_nearest_neighbors(const Base<InputINDEX_t, InputDISTANCE_t, InputQUERY_t>* ptr, int k, int nthreads) {
    auto n = ptr->nobs();
    NeighborList<INDEX_t, DISTANCE_t> output(n);

#ifndef KNNCOLLE_CUSTOM_PARALLEL
    #pragma omp parallel for num_threads(nthreads)
    for (size_t i = 0; i < n; ++i) {
#else
    KNNCOLLE_CUSTOM_PARALLEL(n, [&](size_t first, size_t last) -> void {
    for (size_t i = first; i < last; ++i) {
#endif        
        if constexpr(std::is_same<INDEX_t, InputINDEX_t>::value && std::is_same<DISTANCE_t, InputDISTANCE_t>::value) {
            output[i] = ptr->find_nearest_neighbors(i, k);
        } else {
            auto current = ptr->find_nearest_neighbors(i, k);
            for (const auto& x : current) {
                output[i].emplace_back(x.first, x.second);
            }
        }
    }
#ifdef KNNCOLLE_CUSTOM_PARALLEL    
    }, nthreads);
#endif

    return output;
}

/**
 * Find the nearest neighbors within a pre-built search index.
 * Here, only the neighbor indices are returned, not the distances.
 *
 * @tparam INDEX_t Integer type for the indices in the output object.
 * @tparam InputINDEX_t Integer type for the indices in the input index.
 * @tparam InputDISTANCE_t Floating point type for the distances in the input index.
 * @tparam QUERY_t Floating point type for the query data in the input index.
 *
 * @param ptr Pointer to a `Base` index.
 * @param k Number of nearest neighbors. 
 * @param nthreads Number of threads to use.
 *
 * @return A vector of vectors of length equal to the number of observations in `ptr->nobs()`.
 * Each vector contains the indices of the `k` nearest neighbors for each observation, sorted by increasing distance.
 */
template<typename INDEX_t = int, typename InputINDEX_t, typename InputDISTANCE_t, typename InputQUERY_t> 
std::vector<std::vector<INDEX_t> > find_nearest_neighbors_index_only(const Base<InputINDEX_t, InputDISTANCE_t, InputQUERY_t>* ptr, int k, int nthreads) {
    auto n = ptr->nobs();
    std::vector<std::vector<INDEX_t> > output(n);

#ifndef KNNCOLLE_CUSTOM_PARALLEL
    #pragma omp parallel for num_threads(nthreads)
    for (size_t i = 0; i < n; ++i) {
#else
    KNNCOLLE_CUSTOM_PARALLEL(n, [&](size_t first, size_t last) -> void {
    for (size_t i = first; i < last; ++i) {
#endif        
        auto current = ptr->find_nearest_neighbors(i, k);
        for (const auto& x : current) {
            output[i].push_back(x.first);
        }
    }
#ifdef KNNCOLLE_CUSTOM_PARALLEL    
    }, nthreads);
#endif

    return output;
}

}

#endif
