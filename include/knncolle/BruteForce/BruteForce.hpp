#ifndef KNNCOLLE_BRUTEFORCE_HPP
#define KNNCOLLE_BRUTEFORCE_HPP

#include "../utils/distances.hpp"
#include "../utils/NeighborQueue.hpp"
#include "../utils/Base.hpp"

#include <vector>
#include <type_traits>

/**
 * @file BruteForce.hpp
 *
 * @brief Implements a brute-force search for nearest neighbors.
 */

namespace knncolle {

/**
 * @brief Perform a brute-force nearest neighbor search.
 *
 * The brute-force search computes all pairwise distances between data and query points to identify nearest neighbors of the latter.
 * It has quadratic complexity and is theoretically the worst-performing method;
 * however, it has effectively no overhead from constructing or querying indexing structures, 
 * potentially making it faster in cases where indexing provides little benefit (e.g., few data points, high dimensionality).
 *
 * @tparam DISTANCE Class to compute the distance between vectors, see `distance::Euclidean` for an example.
 * @tparam INDEX_t Integer type for the indices.
 * @tparam DISTANCE_t Floating point type for the distances.
 * @tparam QUERY_t Floating point type for the query data.
 * @tparam INTERNAL_t Floating point type for the internal calculations.
 */
template<class DISTANCE, typename INDEX_t = int, typename DISTANCE_t = double, typename QUERY_t = DISTANCE_t, typename INTERNAL_t = double>
class BruteForce : public Base<INDEX_t, DISTANCE_t, QUERY_t> {
private:
    INDEX_t num_dim;
    INDEX_t num_obs;

public:
    INDEX_t nobs() const { return num_obs; } 
    
    INDEX_t ndim() const { return num_dim; }

private:
    std::vector<INTERNAL_t> store;

public:
    /**
     * @param ndim Number of dimensions.
     * @param nobs Number of observations.
     * @param vals Pointer to an array of length `ndim * nobs`, corresponding to a dimension-by-observation matrix in column-major format, 
     * i.e., contiguous elements belong to the same observation.
     *
     * @tparam INPUT Floating-point type of the input data.
     */
    template<typename INPUT>
    BruteForce(INDEX_t ndim, INDEX_t nobs, const INPUT* vals) : num_dim(ndim), num_obs(nobs), store(vals, vals + ndim * nobs) {}

    std::vector<std::pair<INDEX_t, DISTANCE_t> > find_nearest_neighbors(INDEX_t index, int k) const {
        NeighborQueue<INDEX_t, INTERNAL_t> nearest(k, index);
        search_nn(store.data() + index * num_dim, nearest);

        auto output = nearest.template report<DISTANCE_t>();
        normalize(output);
        return output;
    }

    std::vector<std::pair<INDEX_t, DISTANCE_t> > find_nearest_neighbors(const QUERY_t* query, int k) const {
        NeighborQueue<INDEX_t, INTERNAL_t> nearest(k);
        search_nn(query, nearest);
        auto output = nearest.template report<DISTANCE_t>();
        normalize(output);
        return output;
    }

    const QUERY_t* observation(INDEX_t index, QUERY_t* buffer) const {
        auto candidate = store.data() + num_dim * index;
        if constexpr(std::is_same<QUERY_t, INTERNAL_t>::value) {
            return candidate;
        } else {
            std::copy(candidate, candidate + num_dim, buffer);
            return buffer;
        }
    }

    using Base<INDEX_t, DISTANCE_t, QUERY_t>::observation;

private:
    template<class QUEUE>
    void search_nn(const QUERY_t* query, QUEUE& nearest) const {
        auto copy = store.data();
        for (INDEX_t i = 0; i < num_obs; ++i, copy += num_dim) {
            nearest.add(i, DISTANCE::template raw_distance<INTERNAL_t>(query, copy, num_dim));
        }
        return;
    }

    void normalize(std::vector<std::pair<INDEX_t, DISTANCE_t> >& results) const {
        for (auto& d : results) {
            d.second = DISTANCE::normalize(d.second);
        }
        return;
    } 
};

/**
 * Perform a brute-force search with Euclidean distances.
 */
template<typename INDEX_t = int, typename DISTANCE_t = double, typename QUERY_t = DISTANCE_t, typename INTERNAL_t = double>
using BruteForceEuclidean = BruteForce<distances::Euclidean, INDEX_t, DISTANCE_t, QUERY_t, INTERNAL_t>;

/**
 * Perform a brute-force search with Manhattan distances.
 */
template<typename INDEX_t = int, typename DISTANCE_t = double, typename QUERY_t = DISTANCE_t, typename INTERNAL_t = double>
using BruteForceManhattan = BruteForce<distances::Manhattan, INDEX_t, DISTANCE_t, QUERY_t, INTERNAL_t>;

}

#endif
