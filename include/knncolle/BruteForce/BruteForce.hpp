#ifndef KNNCOLLE_BRUTEFORCE_HPP
#define KNNCOLLE_BRUTEFORCE_HPP

#include "../utils/distances.hpp"
#include "../utils/NeighborQueue.hpp"
#include "../utils/MatrixStore.hpp"
#include "../utils/knn_base.hpp"

#include <vector>

namespace knncolle {

/**
 * @brief Perform a brute-force nearest neighbor search.
 *
 * Implements a brute-force search, mostly for testing purposes.
 * It may also be more performant than the other algorithms for very small numbers of observations.
 *
 * @tparam COPY Whether to copy the input data.
 * @tparam DISTANCE Class to compute the distance between vectors, see `distance::Euclidean` for an example.
 */
template<bool COPY, class DISTANCE>
class BruteForce : public knn_base {
private:
    MatDim_t num_dim;
    CellIndex_t num_obs;

public:
    CellIndex_t nobs() const { return num_obs; } 
    
    MatDim_t ndims() const { return num_dim; }

private:
    MatrixStore<COPY> store;

public:
    BruteForce(CellIndex_t nobs, MatDim_t ndim, const double* vals) : num_dim(ndim), num_obs(nobs), store(ndim * nobs, vals) {}

    bool find_nearest_neighbors(CellIndex_t index, NumNeighbors_t k, std::vector<CellIndex_t>& indices, std::vector<double>& distances, 
        bool report_indices = true, bool report_distances = true, bool check_ties = true) const
    {
        assert(index < static_cast<CellIndex_t>(num_obs));
        NeighborQueue nearest(index, k, check_ties);
        return find_nearest_neighbors_internal(store.reference + index * num_dim, nearest, indices, distances, report_indices, report_distances);
    }

    bool find_nearest_neighbors(const double* query, NumNeighbors_t k, std::vector<CellIndex_t>& indices, std::vector<double>& distances, 
        bool report_indices = true, bool report_distances = true, bool check_ties = true) const
    {
        NeighborQueue nearest(k, check_ties);
        return find_nearest_neighbors_internal(query, nearest, indices, distances, report_indices, report_distances);
    }

private:
    bool find_nearest_neighbors_internal(const double* query, NeighborQueue& nearest, std::vector<CellIndex_t>& indices, std::vector<double>& distances,
        bool report_indices, bool report_distances) const 
    {
        auto copy = store.reference;
        for (CellIndex_t i = 0; i < num_obs; ++i, copy += num_dim) {
            nearest.add(i, DISTANCE::raw_distance(query, copy, num_dim));
        }

        bool out = nearest.report(indices, distances, report_indices, report_distances);
        for (auto& d : distances) {
            d = DISTANCE::normalize(d);
        }

        return out;
    }
};

template<bool COPY = false>
using BruteForceEuclidean = BruteForce<COPY, distances::Euclidean>;

template<bool COPY = false>
using BruteForceManhattan = BruteForce<COPY, distances::Manhattan>;

}

#endif
