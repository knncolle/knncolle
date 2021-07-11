#ifndef KNNCOLLE_ANNOYBASE_HPP
#define KNNCOLLE_ANNOYBASE_HPP

#include <cstdint>
#include "../utils/knn_base.hpp"
#include "annoy/annoylib.h"
#include "annoy/kissrandom.h"

/**
 * @file Annoy.hpp
 *
 * @brief Implements an approximate nearest neighbor search with Annoy.
 */

namespace knncolle {

namespace AnnoyDefaults {

/**
 * Default value for `ntrees` in the `Annoy` constructor.
 */
const int ntrees = 50;

/**
 * Default value for `search_mult` in the `Annoy` constructor.
 */
const double search_mult = -1;

}

/**
 * @brief Perform an approximate nearest neighbor search with Annoy.
 *
 * In the Approximate Nearest Neighbors Oh Yeah (Annoy) algorithm, a tree is constructed where a random hyperplane splits the points into two subsets at each internal node.
 * Leaf nodes are defined when the number of points in a subset falls below a threshold (close to twice the number of dimensions for the settings used here).
 * Multiple trees are constructed in this manner, each of which is different due to the random choice of hyperplanes.
 * For a given query point, each tree is searched to identify the subset of all points in the same leaf node as the query point. 
 * The union of these subsets across all trees is exhaustively searched to identify the actual nearest neighbors to the query.
 *
 * @see
 * Bernhardsson E (2018).
 * Annoy.
 * https://github.com/spotify/annoy
 *
 * @tparam DISTANCE An **Annoy**-derived class to compute the distance between vectors.
 * Note that this is not the same as the classes in `distances.hpp`.
 * @tparam INDEX_t Integer type for the indices.
 * @tparam DISTANCE_t Floating point type for the distances.
 * @tparam INTERNAL_INDEX_t Integer type for the internal indices.
 * @tparam INTERNAL_DATA_t Floating point type for the internal data store.
 * This uses a `float` instead of a `double` to sacrifice some accuracy for performance.
 */
template<class DISTANCE, typename INDEX_t = int, typename DISTANCE_t = double, typename QUERY_t = double, typename INTERNAL_INDEX_t = int32_t, typename INTERNAL_DATA_t = float>
class AnnoySearch : public knn_base<INDEX_t, DISTANCE_t, QUERY_t> {
public:
    INDEX_t nobs() const {
        return annoy_index.get_n_items();
    }
    
    INDEX_t ndim() const {
        return num_dim;
    }

public:
    /**
     * @param ndim Number of dimensions.
     * @param nobs Number of observations.
     * @param vals Pointer to an array of length `ndim * nobs`, corresponding to a dimension-by-observation matrix in column-major format, 
     * i.e., contiguous elements belong to the same observation.
     * @param ntrees Number of trees to construct.
     * Larger values improve accuracy at the cost of index size (i.e., memory usage), see [here](https://github.com/spotify/annoy#tradeoffs) for details.
     * @param search_mult Factor that is multiplied by the number of neighbors `k` to determine the number of nodes to search in `find_nearest_neighbors()`.
     * Larger values improve accuracy at the cost of runtime, see [here](https://github.com/spotify/annoy#tradeoffs) for details.
     * If set to -1, it defaults to `ntrees`.
     *
     * @tparam INPUT Floating-point type of the input data.
     */
    template<typename INPUT>
    AnnoySearch(INDEX_t ndim, INDEX_t nobs, const INPUT* vals, int ntrees = AnnoyDefaults::ntrees, double search_mult = AnnoyDefaults::search_mult) : 
        annoy_index(ndim), num_dim(ndim), search_k_mult(search_mult) 
    {
        if constexpr(std::is_same<INPUT, INTERNAL_DATA_t>::value) {
            for (INDEX_t i=0; i < nobs; ++i, vals += ndim) {
                annoy_index.add_item(i, vals);
            }
        } else {
            std::vector<INTERNAL_DATA_t> incoming(ndim);
            for (INDEX_t i=0; i < nobs; ++i, vals += ndim) {
                std::copy(vals, vals + ndim, incoming.begin());
                annoy_index.add_item(i, incoming.data());
            }
        }
        annoy_index.build(ntrees);
        return;
    }

    std::vector<std::pair<INDEX_t, DISTANCE_t> > find_nearest_neighbors(INDEX_t index, int k) const {
        auto pairs = annoy_index.get_nns_by_item(index, k + 1, get_search_k(k + 1), true); // +1, as it forgets to discard 'self'.
        return reformat(pairs, true, index);
    }
        
    std::vector<std::pair<INDEX_t, DISTANCE_t> > find_nearest_neighbors(const QUERY_t* query, int k) const {
        if constexpr(std::is_same<INTERNAL_DATA_t, QUERY_t>::value) {
            auto pairs = annoy_index.get_nns_by_vector(query, k, get_search_k(k), true);
            return reformat(pairs);
        } else {
            std::vector<INTERNAL_DATA_t> tmp(query, query + num_dim);
            auto pairs = annoy_index.get_nns_by_vector(tmp.data(), k, get_search_k(k), true);
            return reformat(pairs);
        }
    }

private:
    Annoy::AnnoyIndex<INTERNAL_INDEX_t, INTERNAL_DATA_t, DISTANCE, Annoy::Kiss64Random, Annoy::AnnoyIndexSingleThreadedBuildPolicy> annoy_index;
    INDEX_t num_dim;
    double search_k_mult;

    int get_search_k(int k) const {
        if (search_k_mult < 0) {
            return -1;
        } else {
            return search_k_mult * k + 0.5; // rounded up.
        }
    }

    template<class PAIRED> 
    static std::vector<std::pair<INDEX_t, DISTANCE_t> > reformat(const PAIRED& results, bool check_self = false, INDEX_t self_index = 0) {
        // If check_self=false, then we pretend that the self is already
        // found, in which case the two clauses below are just skipped.
        bool self_found = !check_self;

        std::vector<std::pair<INDEX_t, DISTANCE_t> > output;
        for (const auto& current : results) {
            if (!self_found && current.second == self_index) {
                self_found=true;
                continue;
            } else {
                output.push_back(std::pair<INDEX_t, DISTANCE_t>(current.second, current.first));
            }
        }

        // Just in case we're full of ties at duplicate points, such that 'c'
        // is not in the set.  Note that, if self_found=false, we must have at
        // least 'K+2' points for 'c' to not be detected as its own neighbor.
        // Thus there is no need to worry whether we are popping off a non-'c'
        // element at the end of the vector.
        if (!self_found) {
            output.pop_back();
        }
        return output;
    }
};

/**
 * Perform an Annoy search with Euclidean distances.
 */
template<typename INDEX_t = int, typename DISTANCE_t = double, typename QUERY_t = double, typename INTERNAL_INDEX_t = int32_t, typename INTERNAL_DATA_t = float>
using AnnoyEuclidean = AnnoySearch<Annoy::Euclidean, INDEX_t, DISTANCE_t, QUERY_t, INTERNAL_INDEX_t, INTERNAL_DATA_t>;

/**
 * Perform an Annoy search with Manhattan distances.
 */
template<typename INDEX_t = int, typename DISTANCE_t = double, typename QUERY_t = double, typename INTERNAL_INDEX_t = int32_t, typename INTERNAL_DATA_t = float>
using AnnoyManhattan = AnnoySearch<Annoy::Manhattan, INDEX_t, DISTANCE_t, QUERY_t, INTERNAL_INDEX_t, INTERNAL_DATA_t>;

}

#endif
