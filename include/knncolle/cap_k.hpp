#ifndef KNNCOLLE_CAP_K_HPP
#define KNNCOLLE_CAP_K_HPP

#include "sanisizer/sanisizer.hpp"

/**
 * @file cap_k.hpp
 *
 * @brief Cap the number of requested neighbors. 
 */

namespace knncolle {

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
    if (sanisizer::is_less_than(k, num_observations)) {
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
    return sanisizer::min(k, num_observations);
}

}

#endif
