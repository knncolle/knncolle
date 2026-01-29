#ifndef KNNCOLLE_FIND_NEAREST_NEIGHBORS_HPP
#define KNNCOLLE_FIND_NEAREST_NEIGHBORS_HPP

#include "Prebuilt.hpp"

#include <vector>
#include <utility>
#include <type_traits>

#include "sanisizer/sanisizer.hpp"
#ifndef KNNCOLLE_CUSTOM_PARALLEL
#include "subpar/subpar.hpp"
#endif


/**
 * @file find_nearest_neighbors.hpp
 *
 * @brief Find nearest neighbors from an existing index.
 */

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
 * @tparam Distance_ Numeric type for the distances, usually floating-point.
 */
template<typename Index_, typename Distance_>
using NeighborList = std::vector<std::vector<std::pair<Index_, Distance_> > >;

/**
 * Cap the number of neighbors to use in `Searcher::search()` with an index `i`.
 *
 * @tparam Index_ Integer type for the number of observations.
 * @param k Number of nearest neighbors, should be non-negative.
 * @param num_observations Number of observations in the dataset.
 *
 * @return Capped number of neighbors to search for.
 * This is equal to `k` if it is less than `num_observations`;
 * otherwise it is equal to `num_observations - 1` if `num_observations > 0`;
 * otherwise it is equal to zero.
 */
template<typename Index_>
int cap_k(int k, Index_ num_observations) {
    if (sanisizer::is_less_than(sanisizer::attest_gez(k), sanisizer::attest_gez(num_observations))) {
        return k;
    } else if (num_observations) {
        return num_observations - 1;
    } else {
        return 0;
    }
}

/**
 * Cap the number of neighbors to use in `Searcher::search()` with a pointer `query`.
 *
 * @tparam Index_ Integer type for the number of observations.
 * @param k Number of nearest neighbors, should be non-negative.
 * @param num_observations Number of observations in the dataset.
 *
 * @return Capped number of neighbors to query.
 * This is equal to the smaller of `k` and `num_observations`.
 */
template<typename Index_>
int cap_k_query(int k, Index_ num_observations) {
    return sanisizer::min(sanisizer::attest_gez(k), sanisizer::attest_gez(num_observations));
}

/**
 * Find the nearest neighbors within a pre-built index.
 * This is a convenient wrapper around `Searcher::search` that saves the caller the trouble of writing a loop.
 *
 * @tparam Index_ Integer type for the observation indices.
 * @tparam Data_ Numeric type for the input data.
 * @tparam Distance_ Numeric type for the distances, usually floating-point.
 *
 * @param index A `Prebuilt` index.
 * @param k Number of nearest neighbors. 
 * This should be non-negative.
 * Explicitly calling `cap_k()` is not necessary as this is done automatically inside this function.
 * @param num_threads Number of threads to use.
 * The parallelization scheme is defined by `parallelize()`.
 *
 * @return A `NeighborList` of length equal to the number of observations in `index`.
 * Each entry contains (up to) the `k` nearest neighbors for each observation, sorted by increasing distance.
 * The `i`-th entry is guaranteed to not contain `i` itself.
 */
template<typename Index_, typename Data_, typename Distance_>
NeighborList<Index_, Distance_> find_nearest_neighbors(const Prebuilt<Index_, Data_, Distance_>& index, int k, int num_threads = 1) {
    const Index_ nobs = index.num_observations();
    k = cap_k(k, nobs);
    auto output = sanisizer::create<NeighborList<Index_, Distance_> >(sanisizer::attest_gez(nobs));

    parallelize(num_threads, nobs, [&](int, Index_ start, Index_ length) -> void {
        auto sptr = index.initialize_known();
        std::vector<Index_> indices;
        std::vector<Distance_> distances;
        for (Index_ i = start, end = start + length; i < end; ++i) {
            sptr->search(i, k, &indices, &distances);
            const auto actual_k = indices.size();
            output[i].reserve(actual_k);
            for (I<decltype(actual_k)> j = 0; j < actual_k; ++j) {
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
 * @tparam Index_ Integer type for the indices.
 * @tparam Data_ Numeric type for the input and query data.
 * @tparam Distance_ Numeric type for the distances, usually floating-point.
 *
 * @param index A `Prebuilt` index.
 * @param k Number of nearest neighbors. 
 * This should be non-negative.
 * Explicitly calling `cap_k()` is not necessary as this is done automatically inside this function.
 * @param num_threads Number of threads to use.
 * The parallelization scheme is defined by `parallelize()`.
 *
 * @return A vector of vectors of length equal to the number of observations in `index`.
 * Each vector contains the indices of (up to) the `k` nearest neighbors for each observation, sorted by increasing distance.
 * The `i`-th entry is guaranteed to not contain `i` itself.
 */
template<typename Index_, typename Data_, typename Distance_>
std::vector<std::vector<Index_> > find_nearest_neighbors_index_only(const Prebuilt<Index_, Data_, Distance_>& index, int k, int num_threads = 1) {
    const Index_ nobs = index.num_observations();
    k = cap_k(k, nobs);
    auto output = sanisizer::create<std::vector<std::vector<Index_> > >(sanisizer::attest_gez(nobs));

    parallelize(num_threads, nobs, [&](int, Index_ start, Index_ length) -> void {
        auto sptr = index.initialize_known();
        for (Index_ i = start, end = start + length; i < end; ++i) {
            sptr->search(i, k, &(output[i]), NULL);
        }
    });

    return output;
}

}

#endif
