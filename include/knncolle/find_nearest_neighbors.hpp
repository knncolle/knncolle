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

#ifndef KNNCOLLE_CUSTOM_PARALLEL
#include "subpar/subpar.hpp"
#endif

namespace knncolle {

/**
 * @tparam Task_ Integer type for the number of tasks.
 * @tparam Run_ Function to execute a range of tasks.
 *
 * @param num_workers Number of workers.
 * @param num_tasks Number of tasks.
 * @param run_task_range Function to iterate over a range of tasks within a worker.
 *
 * By default, this is an alias to `subpar::parallelize_range()`.
 * However, if the `KNNCOLLE_CUSTOM_PARALLEL` function-like macro is defined, it is called instead. 
 * Any user-defined macro should accept the same arguments as `subpar::parallelize_range()`.
 */
template<typename Task_, class Run_>
void parallelize(int num_workers, Task_ num_tasks, Run_ run_task_range) {
#ifndef KNNCOLLE_CUSTOM_PARALLEL
    // Don't make this nothrow_ = true, as the derived methods could do anything...
    subpar::parallelize(num_workers, num_tasks, std::move(run_task_range));
#else
    KNNCOLLE_CUSTOM_PARALLEL(num_workers, num_tasks, run_task_range);
#endif
}

/**
 * List of nearest neighbors for multiple observations.
 * Each entry corresponds to an observation and contains a nested list (i.e., vector) of its neighbors.
 * Each entry of the nested vector is a pair that contains the identity of the neighbor as an observation index (first) and the distance from the observation to the neighbor (second),
 * sorted by increasing distance.
 *
 * @tparam Index_ Integer type for the indices.
 * @tparam Float_ Floating point type for the distances.
 */
template<typename Index_ = int, typename Float_ = double> 
using NeighborList = std::vector<std::vector<std::pair<Index_, Float_> > >;

/**
 * Find the nearest neighbors within a pre-built index.
 * This is a convenient wrapper around `Searcher::search` that saves the caller the trouble of writing a loop.
 *
 * @tparam Dim_ Integer type for the number of dimensions.
 * @tparam Index_ Integer type for the indices.
 * @tparam Float_ Floating point type for the query data and output distances.
 *
 * @param index A `Prebuilt` index.
 * @param k Number of nearest neighbors. 
 * @param num_threads Number of threads to use.
 * The parallelization scheme is defined by `parallelize()`.
 *
 * @return A `NeighborList` of length equal to the number of observations in `index`.
 * Each entry contains (up to) the `k` nearest neighbors for each observation, sorted by increasing distance.
 * The `i`-th entry is guaranteed to not contain `i` itself.
 */
template<typename Dim_, typename Index_, typename Float_>
NeighborList<Index_, Float_> find_nearest_neighbors(const Prebuilt<Dim_, Index_, Float_>& index, int k, int num_threads = 1) {
    Index_ nobs = index.num_observations();
    NeighborList<Index_, Float_> output(nobs);

    parallelize(num_threads, nobs, [&](int, Index_ start, Index_ length) -> void {
        auto sptr = index.initialize();
        std::vector<Index_> indices;
        std::vector<Float_> distances;
        for (Index_ i = start, end = start + length; i < end; ++i) {
            sptr->search(i, k, &indices, &distances);
            int actual_k = indices.size();
            output[i].reserve(actual_k);
            for (int j = 0; j < actual_k; ++j) {
                output[i].emplace_back(indices[j], distances[j]);
            }
        }
    });

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
 * The parallelization scheme is defined by `parallelize()`.
 *
 * @return A vector of vectors of length equal to the number of observations in `index`.
 * Each vector contains the indices of (up to) the `k` nearest neighbors for each observation, sorted by increasing distance.
 * The `i`-th entry is guaranteed to not contain `i` itself.
 */
template<typename Dim_, typename Index_, typename Float_>
std::vector<std::vector<Index_> > find_nearest_neighbors_index_only(const Prebuilt<Dim_, Index_, Float_>& index, int k, int num_threads = 1) {
    Index_ nobs = index.num_observations();
    std::vector<std::vector<Index_> > output(nobs);

    parallelize(num_threads, nobs, [&](int, Index_ start, Index_ length) -> void {
        auto sptr = index.initialize();
        for (Index_ i = start, end = start + length; i < end; ++i) {
            sptr->search(i, k, &(output[i]), NULL);
        }
    });

    return output;
}

}

#endif
