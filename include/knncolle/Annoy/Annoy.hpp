#ifndef KNNCOLLE_ANNOYBASE_HPP
#define KNNCOLLE_ANNOYBASE_HPP

#include "../utils/knn_base.hpp"
#include "annoy/annoylib.h"
#include "annoy/kissrandom.h"

/**
 * @file Annoy.hpp
 *
 * Implements an approximate nearest neighbor search with Annoy.
 */

namespace knncolle {

/**
 * @brief Perform an approximate nearest neighbor search with Annoy.
 *
 * @tparam DISTANCE An **Annoy**-derived class to compute the distance between vectors.
 * Note that this is not the same as the classes in `distances.hpp`.
 * @tparam ITYPE Integer type for the indices.
 * @tparam DTYPE Floating point type for the data.
 */
template<class DISTANCE, typename ITYPE = int, typename DTYPE = double>
class AnnoySearch : public knn_base<ITYPE, DTYPE> {
public:
    ITYPE nobs() const {
        return annoy_index.get_n_items();
    }
    
    ITYPE ndim() const {
        return num_dim;
    }

public:
    /**
     * Construct an `Annoy` instance.
     *
     * @param ndim Number of dimensions.
     * @param nobs Number of observations.
     * @param vals Pointer to an array of length `ndim * nobs`, corresponding to a dimension-by-observation matrix in column-major format, 
     * i.e., contiguous elements belong to the same observation.
     */
    AnnoySearch(ITYPE ndim, ITYPE nobs, const DTYPE* vals, int ntrees = 50, double search_mult = 50) : annoy_index(ndim), num_dim(ndim), search_k_mult(search_mult) {
        for (ITYPE i=0; i < nobs; ++i, vals += ndim) {
            annoy_index.add_item(i, vals);
        }
        annoy_index.build(ntrees);
        return;
    }

    void find_nearest_neighbors(ITYPE index, int k, std::vector<ITYPE>* indices, std::vector<DTYPE>* distances) const { 
        if (distances) {
            distances->clear();
        }

        if (!indices) {
            std::vector<ITYPE> tmp_indices;
            annoy_index.get_nns_by_item(index, k + 1, get_search_k(k + 1), &tmp_indices, distances); // +1, as it forgets to discard 'self'.
            purge_self(index, &tmp_indices, distances);
        } else {
            indices->clear();
            annoy_index.get_nns_by_item(index, k + 1, get_search_k(k + 1), indices, distances); 
            purge_self(index, indices, distances);
        }

        return;
    }
        
    void find_nearest_neighbors(const DTYPE* query, int k, std::vector<ITYPE>* indices, std::vector<DTYPE>* distances) const {
        if (distances) {
            distances->clear();
        }

        if (!indices) {
            std::vector<ITYPE> tmp_indices;
            annoy_index.get_nns_by_vector(query, k, get_search_k(k), &tmp_indices, distances);
        } else {
            indices->clear();
            annoy_index.get_nns_by_vector(query, k, get_search_k(k), indices, distances);
        }
        return;
    }

private:
    Annoy::AnnoyIndex<ITYPE, DTYPE, DISTANCE, Annoy::Kiss64Random, Annoy::AnnoyIndexSingleThreadedBuildPolicy> annoy_index;
    ITYPE num_dim;
    double search_k_mult;

    int get_search_k(int k) const {
        return search_k_mult * k + 0.5; // rounded up.
    }

    static void purge_self(ITYPE self_index, std::vector<ITYPE>* indices, std::vector<DTYPE>* distances) {
        bool self_found=false;
        auto& current = *indices;
        for (size_t idx=0; idx < current.size(); ++idx) {
            if (current[idx] == self_index) {
                current.erase(current.begin() + idx);
                if (distances) {
                    distances->erase(distances->begin() + idx);
                }
                self_found=true;
                break;
            }
        }

        // Just in case we're full of ties at duplicate points, such that 'c' is not in the set.
        // Note that, if self_found=false, we must have at least 'K+2' points for 'c' to not 
        // be detected as its own neighbor. Thus there is no need to worry whether we are popping 
        // off a non-'c' element at the end of the vector.
        if (!self_found) {
            current.pop_back();
            if (distances) {
                distances->pop_back();
            }
        }
    }
};

/**
 * Perform an Annoy search with Euclidean distances.
 */
template<typename ITYPE = int, typename DTYPE = double>
using AnnoyEuclidean = AnnoySearch<Annoy::Euclidean, ITYPE, DTYPE>;

/**
 * Perform an Annoy search with Manhattan distances.
 */
template<typename ITYPE = int, typename DTYPE = double>
using AnnoyManhattan = AnnoySearch<Annoy::Manhattan, ITYPE, DTYPE>;

}

#endif
