#ifndef KNNCOLLE_REPORT_ALL_NEIGHBORS_HPP
#define KNNCOLLE_REPORT_ALL_NEIGHBORS_HPP

#include <algorithm>
#include <vector>

/**
 * @file report_all_neighbors.hpp
 * @brief Format the output for `Searcher::search_all()`.
 */

namespace knncolle {

/**
 * Count the number of neighbors from a range-based search, after removing the observation being searched.
 * This is intended for developer use in implementations of `Searcher::search_all()`,
 * and protects against pathological situations where an observation fails to be detected as its own neighbor.
 *
 * @param count Number of neighbors within range of the specified observation, including the observation itself.
 * @return `count - 1` in most cases, otherwise zero.
 */
template<typename Index_>
Index_ count_all_neighbors_without_self(Index_ count) {
    return (count ? count - 1 : 0);
}

/**
 * @cond
 */
namespace internal {

template<bool do_indices_, bool do_distances_, typename Distance_, typename Index_>
void report_all_neighbors_raw(std::vector<std::pair<Distance_, Index_> >& all_neighbors, std::vector<Index_>* output_indices, std::vector<Distance_>* output_distances, Index_ i) {
    std::sort(all_neighbors.begin(), all_neighbors.end());

    auto target_size = count_all_neighbors_without_self(all_neighbors.size());
    if constexpr(do_indices_) {
        output_indices->clear();
        output_indices->reserve(target_size);
    }
    if constexpr(do_distances_) {
        output_distances->clear();
        output_distances->reserve(target_size);
    }

    for (const auto& an : all_neighbors) {
        if (an.second != i) {
            if constexpr(do_indices_) {
                output_indices->push_back(an.second);
            }
            if constexpr(do_distances_) {
                output_distances->push_back(an.first);
            }
        }
    }
}

template<bool do_indices_, bool do_distances_, typename Distance_, typename Index_>
void report_all_neighbors_raw(std::vector<std::pair<Distance_, Index_> >& all_neighbors, std::vector<Index_>* output_indices, std::vector<Distance_>* output_distances) {
    std::sort(all_neighbors.begin(), all_neighbors.end());

    auto target_size = all_neighbors.size();
    if constexpr(do_indices_) {
        output_indices->clear();
        output_indices->reserve(target_size);
    }
    if constexpr(do_distances_) {
        output_distances->clear();
        output_distances->reserve(target_size);
    }

    for (const auto& an : all_neighbors) {
        if constexpr(do_indices_) {
            output_indices->push_back(an.second);
        }
        if constexpr(do_distances_) {
            output_distances->push_back(an.first);
        }
    }
}

}
/**
 * @endcond
 */

/**
 * Report the indices and distances of all neighbors in range of an observation of interest.
 * If the observation of interest is detected as its own neighbor, it will be removed from the output vectors.
 *
 * @tparam Distance_ Floating point type for the distances.
 * @tparam Index_ Integer type for the observation indices.
 *
 * @param[in] all_neighbors Vector of (distance, index) pairs for all neighbors within range of the observation of interest.
 * This may include the observation itself.
 * Note that this will be sorted on output.
 * @param[out] output_indices Pointer to a vector in which to store the indices of all neighbors in range, sorted by distance.
 * If `NULL`, the indices will not be reported.
 * @param[out] output_distances Pointer to a vector in which to store the (sorted) distances to all neighbors in range.
 * If `NULL`, the distances will not be reported.
 * Otherwise, on output, this will have the same length as `*output_indices` and contain distances to each of those neighbors.
 * @param self Index of the observation of interest, i.e., for which neighbors are to be identified.
 * If present in `all_neighbors`, this will be removed from `output_indices` and `output_distances`.
 */
template<typename Distance_, typename Index_>
void report_all_neighbors(std::vector<std::pair<Distance_, Index_> >& all_neighbors, std::vector<Index_>* output_indices, std::vector<Distance_>* output_distances, Index_ self) {
    if (output_indices && output_distances) {
        internal::report_all_neighbors_raw<true, true>(all_neighbors, output_indices, output_distances, self);
    } else if (output_indices) {
        internal::report_all_neighbors_raw<true, false>(all_neighbors, output_indices, output_distances, self);
    } else if (output_distances) {
        internal::report_all_neighbors_raw<false, true>(all_neighbors, output_indices, output_distances, self);
    }
}

/**
 * Report the indices and distances of all neighbors in range of an observation of interest.
 * It is assumed that the observation of interest is not detected as its own neighbor, presumably as it does not exist in the original input dataset.
 *
 * @tparam Distance_ Floating point type for the distances.
 * @tparam Index_ Integer type for the observation indices.
 *
 * @param[in] all_neighbors Vector of (distance, index) pairs for all neighbors within range of the observation of interest.
 * This may include the observation itself.
 * Note that this will be sorted on output.
 * @param[out] output_indices Pointer to a vector in which to store the indices of all neighbors in range, sorted by distance.
 * If `NULL`, the indices will not be reported.
 * @param[out] output_distances Pointer to a vector in which to store the (sorted) distances to all neighbors in range.
 * If `NULL`, the distances will not be reported.
 * Otherwise, on output, this will have the same length as `*output_indices` and contain distances to each of those neighbors.
 */
template<typename Distance_, typename Index_>
void report_all_neighbors(std::vector<std::pair<Distance_, Index_> >& all_neighbors, std::vector<Index_>* output_indices, std::vector<Distance_>* output_distances) {
    if (output_indices && output_distances) {
        internal::report_all_neighbors_raw<true, true>(all_neighbors, output_indices, output_distances);
    } else if (output_indices) {
        internal::report_all_neighbors_raw<true, false>(all_neighbors, output_indices, output_distances);
    } else if (output_distances) {
        internal::report_all_neighbors_raw<false, true>(all_neighbors, output_indices, output_distances);
    }
}

}

#endif
