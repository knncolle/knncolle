#ifndef KNNCOLLE_FIND_NEAREST_NEIGHBORS_HPP
#define KNNCOLLE_FIND_NEAREST_NEIGHBORS_HPP

#include <vector>
#include <utility>
#include "Prebuilt.hpp"

/**
 * @file find_nearest_neighbors.hpp
 *
 * @brief Find nearest neighbors from an existing index.
 */

namespace knncolle {

/**
 * List of nearest neighbors for multiple observations.
 * Each entry corresponds to an observation and contains a pair of vectors.
 * The first vector contains the identities of the nearest neighbors of the observation, sorted by increasing distance,
 * while the second vector contains the distances to those neighbors.
 *
 * @tparam Index_ Integer type for the indices.
 * @tparam Float_ Floating point type for the distances.
 */
template<typename Index_ = int, typename Float_ = double> 
using NeighborList = std::vector<std::pair<std::vector<Index_>, std::vector<Float_> > >;

/**
 * Find the nearest neighbors within a pre-built index.
 * This is a convenient wrapper around `Searcher::search` that saves the caller the trouble of writing a loop.
 *
 * Advanced users can define a `KNNCOLLE_CUSTOM_PARALLEL` function-like macro.
 * This will be passed three arguments - the number of observations, the number of threads,
 * and a function `fun` that accepts a start and length of a contiguous block of observations.
 * The `KNNCOLLE_CUSTOM_PARALLEL` call is responsible for partitioning the observations into contiguous blocks that are assigned to each thread;
 * the `fun` should be called on each block to find the neighbors for that block.
 *
 * @tparam Dim_ Integer type for the number of dimensions.
 * @tparam Index_ Integer type for the indices.
 * @tparam Float_ Floating point type for the query data and output distances.
 *
 * @param index A `Prebuilt` index.
 * @param k Number of nearest neighbors. 
 * @param num_threads Number of threads to use.
 *
 * @return A `NeighborList` of length equal to the number of observations in `index`.
 * Each entry contains (up to) the `k` nearest neighbors for each observation, sorted by increasing distance.
 * The `i`-th entry is guaranteed to not contain `i` itself.
 */
template<typename Dim_, typename Index_, typename Float_>
NeighborList<Index_, Float_> find_nearest_neighbors(const Prebuilt<Dim_, Index_, Float_>& index, int k, [[maybe_unused]] int num_threads = 1) {
    Index_ nobs = index.num_observations();
    NeighborList<Index_, Float_> output(nobs);

#ifndef KNNCOLLE_CUSTOM_PARALLEL
#ifdef _OPENMP
    #pragma omp parallel num_threads(num_threads)
    {
    auto sptr = index.initialize();
    #pragma omp for
    for (Index_ i = 0; i < nobs; ++i) {
#else
    auto sptr = index.initialize();
    for (Index_ i = 0; i < nobs; ++i) {
#endif
#else
    KNNCOLLE_CUSTOM_PARALLEL(nobs, num_threads, [&](Index_ start, Index_ length) -> void {
    auto sptr = index.initialize();
    for (Index_ i = start, end = start + length; i < end; ++i) {
#endif        

        sptr->search(i, k, &(output[i].first), &(output[i].second));

#ifndef KNNCOLLE_CUSTOM_PARALLEL    
#ifdef _OPENMP
    }
    }
#else
    }
#endif
#else
    }
    });
#endif

    return output;
}

/**
 * Find the nearest neighbors within a pre-built search index.
 * Here, only the neighbor indices are returned, not the distances.
 * This function will also respond to any defined `KNNCOLLE_CUSTOM_PARALLEL`, see `find_nearest_neighbors()` for details.
 *
 * @tparam Dim_ Integer type for the number of dimensions.
 * @tparam Index_ Integer type for the indices.
 * @tparam Float_ Floating point type for the query data and output distances.
 *
 * @param index A `Prebuilt` index.
 * @param k Number of nearest neighbors. 
 * @param num_threads Number of threads to use.
 *
 * @return A vector of vectors of length equal to the number of observations in `index`.
 * Each vector contains the indices of (up to) the `k` nearest neighbors for each observation, sorted by increasing distance.
 * The `i`-th entry is guaranteed to not contain `i` itself.
 */
template<typename Dim_, typename Index_, typename Float_>
std::vector<std::vector<Index_> > find_nearest_neighbors_index_only(const Prebuilt<Dim_, Index_, Float_>& index, int k, [[maybe_unused]] int num_threads = 1) {
    Index_ nobs = index.num_observations();
    std::vector<std::vector<Index_> > output(nobs);

#ifndef KNNCOLLE_CUSTOM_PARALLEL
#ifdef _OPENMP
    #pragma omp parallel num_threads(num_threads)
    {
    auto sptr = index.initialize();
    #pragma omp for
    for (Index_ i = 0; i < nobs; ++i) {
#else
    auto sptr = index.initialize();
    for (Index_ i = 0; i < nobs; ++i) {
#endif
#else
    KNNCOLLE_CUSTOM_PARALLEL(nobs, num_threads, [&](Index_ start, Index_ length) -> void {
    auto sptr = index.initialize();
    for (Index_ i = start, end = start + length; i < end; ++i) {
#endif        

        sptr->search(i, k, &(output[i]), NULL);

#ifndef KNNCOLLE_CUSTOM_PARALLEL
#ifdef _OPENMP
    }
    }
#else
    }
#endif
#else
    }
    });
#endif

    return output;
}

}

#endif
