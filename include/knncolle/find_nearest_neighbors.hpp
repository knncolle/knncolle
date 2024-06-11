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
 * Each entry corresponds to an observation and contains the nearest neighbors as (index, distance) pairs for that observation.
 *
 * @tparam Index_ Integer type for the indices.
 * @tparam Float_ Floating point type for the distances.
 */
template<typename Index_ = int, typename Float_ = double> 
using NeighborList = std::vector<std::vector<std::pair<Index_, Float_> > >;

/**
 * Find the nearest neighbors within a pre-built index.
 * This is a convenient wrapper around `Prebuilt::search` that saves the caller the trouble of writing a loop.
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
 * Each entry contains the `k` nearest neighbors for each observation, sorted by increasing distance.
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

        sptr->search(i, k, output[i]);

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
 * Each vector contains the indices of the `k` nearest neighbors for each observation, sorted by increasing distance.
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
    std::vector<std::pair<Index_, Float_> > tmp;
    #pragma omp for
    for (Index_ i = 0; i < nobs; ++i) {
#else
    auto sptr = index.initialize();
    std::vector<std::pair<Index_, Float_> > tmp;
    for (Index_ i = 0; i < nobs; ++i) {
#endif
#else
    KNNCOLLE_CUSTOM_PARALLEL(nobs, num_threads, [&](Index_ start, Index_ length) -> void {
    auto sptr = index.initialize();
    std::vector<std::pair<Index_, Float_> > tmp;
    for (Index_ i = start, end = start + length; i < end; ++i) {
#endif        

        sptr->search(i, k, tmp);
        for (const auto& x : tmp) {
            output[i].push_back(x.first);
        }

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
