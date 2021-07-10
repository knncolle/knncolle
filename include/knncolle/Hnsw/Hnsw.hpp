#ifndef KNNCOLLE_HNSW_HPP
#define KNNCOLLE_HNSW_HPP

#include "../utils/knn_base.hpp"
#include "../utils/NeighborQueue.hpp"

#include "hnswlib/hnswalg.h"
#include <cmath>

/**
 * @file Hnsw.hpp
 *
 * Implements an approximate nearest neighbor search with HNSW.
 */

namespace knncolle {

/**
 * @brief Perform an approximate nearest neighbor search with HNSW.
 *
 * @tparam DISTANCE An **hnswlib**-derived class to compute the distance between vectors.
 * Note that this is not the same as the classes in `distances.hpp`.
 * @tparam ITYPE Integer type for the indices.
 * @tparam DTYPE Floating point type for the data.
 */
template<class SPACE, typename ITYPE = int, typename DTYPE = double>
class HnswSearch : public knn_base<ITYPE, DTYPE> {
public:
    ITYPE nobs() const {
        return num_obs;
    }
    
    ITYPE ndim() const {
        return num_dim;
    }

public:
    /**
     * Construct an `HnswSearch` instance.
     *
     * @param ndim Number of dimensions.
     * @param nobs Number of observations.
     * @param vals Pointer to an array of length `ndim * nobs`, corresponding to a dimension-by-observation matrix in column-major format, 
     * i.e., contiguous elements belong to the same observation.
     */
    HnswSearch(ITYPE ndim, ITYPE nobs, const DTYPE* vals, int nlinks = 16, int ef_construction= 200, int ef_search = 10) : 
        space(ndim), hnsw_index(&space, nobs, nlinks, ef_construction), num_dim(ndim), num_obs(nobs)
    {
        std::vector<float> copy(ndim);
        for (ITYPE i=0; i < nobs; ++i, vals += ndim) {
            std::copy(vals, vals + ndim, copy.begin());
            hnsw_index.addPoint(copy.data(), i);
        }
        hnsw_index.setEf(ef_search);
        return;
    }

    void find_nearest_neighbors(ITYPE index, int k, std::vector<ITYPE>* indices, std::vector<DTYPE>* distances) const { 
        auto V = hnsw_index.getDataByLabel<float>(index);
        auto Q = hnsw_index.searchKnn(V.data(), k+1);
        harvest_queue(Q, indices, distances, true, index);
        normalize(distances);
        return;
    }
        
    void find_nearest_neighbors(const DTYPE* query, int k, std::vector<ITYPE>* indices, std::vector<DTYPE>* distances) const {
        std::vector<float> copy(query, query + num_dim);
        auto Q = hnsw_index.searchKnn(copy.data(), k);
        harvest_queue(Q, indices, distances);
        normalize(distances);
        return;
    }

private:
    SPACE space;
    hnswlib::HierarchicalNSW<float> hnsw_index;
    ITYPE num_dim, num_obs;

    static void normalize (std::vector<DTYPE>* distances) {
        if (distances) {
            for (auto& d : *distances) {
                d = SPACE::normalize(d);
            }
        }
        return;
    }
};

namespace hnsw_distances {

class Manhattan : public hnswlib::SpaceInterface<float> {
    size_t data_size_;
    size_t dim_;
public:
    Manhattan(size_t ndim) : data_size_(ndim * sizeof(float)), dim_(ndim) {}

    ~Manhattan() {}

    size_t get_data_size() {
        return data_size_;
    }

    hnswlib::DISTFUNC<float> get_dist_func() {
        return L1;
    }

    void * get_dist_func_param() {
        return &dim_;
    }

    static float L1(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        //return *((float*)pVect2);
        const float* pVect1=static_cast<const float*>(pVect1v);
        const float* pVect2=static_cast<const float*>(pVect2v);
        size_t qty = *((size_t *) qty_ptr);
        float res = 0;
        for (; qty > 0; --qty, ++pVect1, ++pVect2) {
            res += std::fabs(*pVect1 - *pVect2);
        }
        return res;
    }
    
    static float normalize(float raw) {
        return raw;
    }
};

class Euclidean : public hnswlib::L2Space {
public:
    Euclidean(size_t ndim) : hnswlib::L2Space(ndim) {}

    static float normalize(float raw) {
        return std::sqrt(raw);
    }
};

}

/**
 * Perform an Hnsw search with Euclidean distances.
 */
template<typename ITYPE = int, typename DTYPE = double>
using HnswEuclidean = HnswSearch<hnsw_distances::Euclidean, ITYPE, DTYPE>;

/**
 * Perform an Hnsw search with Manhattan distances.
 */
template<typename ITYPE = int, typename DTYPE = double>
using HnswManhattan = HnswSearch<hnsw_distances::Manhattan, ITYPE, DTYPE>;

}

#endif
