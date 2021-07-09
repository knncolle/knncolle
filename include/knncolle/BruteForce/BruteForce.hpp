#ifndef KNNCOLLE_BRUTEFORCE_HPP
#define KNNCOLLE_BRUTEFORCE_HPP

#include "../utils/distances.hpp"
#include "../utils/NeighborQueue.hpp"
#include "../utils/MatrixStore.hpp"
#include "../utils/knn_base.hpp"

#include <vector>

/**
 * @file BruteForce.hpp
 *
 * Implements a brute-force search for nearest neighbors.
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
 * @tparam DISTANCE Template class to compute the distance between vectors, see `distance::Euclidean` for an example.
 * @tparam ITYPE Integer type for the indices.
 * @tparam DTYPE Floating point type for the data.
 */
template<template<typename, typename> class DISTANCE, typename ITYPE = int, typename DTYPE = double>
class BruteForce : public knn_base<ITYPE, DTYPE> {
private:
    ITYPE num_dim;
    ITYPE num_obs;

public:
    ITYPE nobs() const { return num_obs; } 
    
    ITYPE ndim() const { return num_dim; }

private:
    MatrixStore<DTYPE> store;

public:
    /**
     * Construct a `BruteForce` instance without any copying of the data.
     * The `vals` pointer is directly stored in the instance, assuming that the lifetime of the array exceeds that of the `BruteForce` object.
     *
     * @param ndim Number of dimensions.
     * @param nobs Number of observations.
     * @param vals Pointer to an array of length `ndim * nobs`, corresponding to a dimension-by-observation matrix in column-major format, 
     * i.e., contiguous elements belong to the same observation.
     */
    BruteForce(ITYPE ndim, ITYPE nobs, const DTYPE* vals) : num_dim(ndim), num_obs(nobs), store(vals) {}

    /**
     * Construct a `BruteForce` instance by copying the data.
     * This is useful when the original data container has an unknown lifetime.
     *
     * @param ndim Number of dimensions.
     * @param nobs Number of observations.
     * @param vals Vector of length `ndim * nobs`, corresponding to a dimension-by-observation matrix in column-major format, 
     * i.e., contiguous elements belong to the same observation.
     */
    BruteForce(ITYPE ndim, ITYPE nobs, std::vector<DTYPE> vals) : num_dim(ndim), num_obs(nobs), store(std::move(vals)) {}

    void find_nearest_neighbors(ITYPE index, int k, std::vector<ITYPE>* indices, std::vector<DTYPE>* distances) const {
        assert(index < num_obs);
        NeighborQueue<ITYPE, DTYPE> nearest(k + 1);
        search_nn(store.reference + index * num_dim, nearest);
        nearest.report(indices, distances, true, index);
        normalize(distances);
        return;
    }

    void find_nearest_neighbors(const DTYPE* query, int k, std::vector<ITYPE>* indices, std::vector<DTYPE>* distances) const {
        NeighborQueue<ITYPE, DTYPE> nearest(k);
        search_nn(query, nearest);
        nearest.report(indices, distances);
        normalize(distances);
        return;
    }

private:
    void search_nn(const DTYPE* query, NeighborQueue<ITYPE, DTYPE>& nearest) const {
        auto copy = store.reference;
        for (ITYPE i = 0; i < num_obs; ++i, copy += num_dim) {
            nearest.add(i, DISTANCE<ITYPE, DTYPE>::raw_distance(query, copy, num_dim));
        }
        return;
    }

    void normalize(std::vector<DTYPE>* distances) const {
        if (distances) {
            for (auto& d : *distances) {
                d = DISTANCE<ITYPE, DTYPE>::normalize(d);
            }
        }
        return;
    }
};

/**
 * Perform a brute-force search with Euclidean distances.
 */
template<typename ITYPE = int, typename DTYPE = double>
using BruteForceEuclidean = BruteForce<distances::Euclidean, ITYPE, DTYPE>;

/**
 * Perform a brute-force search with Manhattan distances.
 */
template<typename ITYPE = int, typename DTYPE = double>
using BruteForceManhattan = BruteForce<distances::Manhattan, ITYPE, DTYPE>;

}

#endif
