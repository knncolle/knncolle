#ifndef KNNCOLLE_BRUTEFORCE_HPP
#define KNNCOLLE_BRUTEFORCE_HPP

#include "../utils/distances.hpp"
#include "../utils/NeighborQueue.hpp"
#include "../utils/MatrixStore.hpp"
#include "../utils/knn_base.hpp"

#include <vector>

namespace knncolle {

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

    void find_nearest_neighbors(CellIndex_t index, NumNeighbors_t k) {
        assert(index < static_cast<CellIndex_t>(num_obs));
        NeighborQueue nearest(index, k, this->get_ties);
        find_nearest_neighbors_internal(store.reference + index * num_dim, nearest);
        return;
    }

    void find_nearest_neighbors(const double* query, NumNeighbors_t k) {
        NeighborQueue nearest(k, this->get_ties);
        find_nearest_neighbors_internal(query, nearest);
        return;
    }

private:
    void find_nearest_neighbors_internal(const double* query, NeighborQueue& nearest) {
        auto copy = store.reference;
        for (CellIndex_t i = 0; i < num_obs; ++i, copy += num_dim) {
            nearest.add(i, DISTANCE::raw_distance(query, copy, num_dim));
        }

        nearest.report(this->current_neighbors, this->current_distances, this->current_tied, this->get_index, this->get_distance);
        if (this->get_distance) {
            for (auto& d : this->current_distances) {
                d = DISTANCE::normalize(d);
            }
        }

        return;
    }
};

template<bool COPY = false>
using BruteForceEuclidean = BruteForce<COPY, distances::Euclidean>;

template<bool COPY = false>
using BruteForceManhattan = BruteForce<COPY, distances::Manhattan>;

}

#endif
