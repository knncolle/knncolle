#ifndef KNNCOLLE_HPP
#define KNNCOLLE_HPP

#include "BruteForce/BruteForce.hpp"
#include "Kmknn/Kmknn.hpp"
#include "VpTree/VpTree.hpp"
#include "Annoy/Annoy.hpp"
#include "Hnsw/Hnsw.hpp"
#include <memory>

/**
 * @file knncolle.hpp
 *
 * @brief Umbrella header to include all algorithms.
 */

namespace knncolle {

/**
 * Distance metrics to use for searching.
 */
enum DispatchDistance { EUCLIDEAN, MANHATTAN };

/**
 * Choice of available KNN algorithms. 
 */
enum DispatchAlgorithm { BRUTEFORCE, KMKNN, VPTREE, ANNOY, HNSW };

/**
 * @brief Top-level helper class to run any available search algorithms.
 *
 * With an instance of a `Dispatch` class, clients can run any KNN search algorithm via `Dispatch::build()`.
 * Optional parameters can be set by modifying the public members of `Dispatch`'s internal structures,
 * e.g., we can control the parameters used in the **Annoy** search by modifying the `Dispatch::Annoy` members.
 *
 * @tparam INDEX_t Integer type for the indices.
 * @tparam DISTANCE_t Floating point type for the distances.
 * @tparam QUERY_t Floating point type for the query data.
 */
template<typename INDEX_t = int, typename DISTANCE_t = double, typename QUERY_t = double>
class Dispatch {
public:
    /**
     * Type of distance metric to use.
     */
    DispatchDistance distance_type = EUCLIDEAN;
public:
    /**
     * @brief Parameter store for the brute-force search.
     */
    struct BruteForce_param {
        /** 
         * See the `BruteForce::BruteForce()` constructor.
         */
        bool copy = false;
    };

    /**
     * Parameters to be passed to the `BruteForce` constructor.
     */
    BruteForce_param BruteForce;

    /**
     * @brief Parameter store for the vantage point tree search.
     */
    struct VpTree_param {
        /** 
         * See the `VpTree::VpTree()` constructor.
         */
        bool copy = false;
    };
    
    /**
     * Parameters to be passed to the `VpTree` constructor.
     */
    VpTree_param VpTree;

    /**
     * @brief Parameter store for the k-means-based search.
     */
    struct Kmknn_param {
        /** 
         * See the `Kmknn::Kmknn()` constructor.
         */
        double power = 0.5;
    };
    
    /**
     * Parameters to be passed to the `Kmknn` constructor.
     */
    Kmknn_param Kmknn;

    /**
     * @brief Parameter store for the Annoy search.
     */
    struct Annoy_param {
        /** 
         * See the `AnnoySearch::AnnoySearch()` constructor.
         */
        bool ntrees = AnnoyDefaults::ntrees;

        /** 
         * See the `AnnoySearch::AnnoySearch()` constructor.
         */
        double search_mult = AnnoyDefaults::search_mult;
    };
    
    /**
     * Parameters to be passed to the `AnnoySearch` constructor.
     */
    Annoy_param Annoy;

    /**
     * @brief Parameter store for the HNSW search.
     */
    struct Hnsw_param {
        /** 
         * See the `Hnsw::Hnsw()` constructor.
         */
        int nlinks = HnswDefaults::nlinks;

        /** 
         * See the `Hnsw::Hnsw()` constructor.
         */
        int ef_construction = HnswDefaults::ef_construction;

        /** 
         * See the `Hnsw::Hnsw()` constructor.
         */
        int ef_search = HnswDefaults::ef_search;
    };
    
    /**
     * Parameters to be passed to the `Hnsw` constructor.
     */
    Hnsw_param Hnsw;

public:
    /**
     * @param ndim Number of dimensions.
     * @param nobs Number of observations.
     * @param vals Pointer to an array of length `ndim * nobs`, corresponding to a dimension-by-observation matrix in column-major format, 
     * i.e., contiguous elements belong to the same observation.
     * @param algorithm Choice of KNN search algorithm.
     *
     * @tparam INPUT Floating-point type of the input data.
     *
     * @return A pointer to a `Base` subclass, which can be used to identify nearest neighbors using the `Base::find_nearest_neighbors()` method.
     * The exact algorithm used depends on the choice of algorithm in `algorithm`.
     */
    template<class INPUT> 
    std::shared_ptr<Base<INDEX_t, DISTANCE_t, QUERY_t> > build(INDEX_t ndim, INDEX_t nobs, const INPUT* vals, DispatchAlgorithm algorithm) {
        typedef Base<INDEX_t, DISTANCE_t, QUERY_t> BASE;
        if (distance_type == EUCLIDEAN) {
            switch(algorithm) {
                case KMKNN:
                    return std::shared_ptr<BASE>(new KmknnEuclidean<INDEX_t, DISTANCE_t, QUERY_t>(ndim, nobs, vals, Kmknn.power));
                case VPTREE:
                    return std::shared_ptr<BASE>(new VpTreeEuclidean<INDEX_t, DISTANCE_t, QUERY_t>(ndim, nobs, vals, VpTree.copy));
                case ANNOY:
                    return std::shared_ptr<BASE>(new AnnoyEuclidean<INDEX_t, DISTANCE_t, QUERY_t>(ndim, nobs, vals, Annoy.ntrees, Annoy.search_mult));
                case HNSW:
                    return std::shared_ptr<BASE>(new HnswEuclidean<INDEX_t, DISTANCE_t, QUERY_t>(ndim, nobs, vals, Hnsw.nlinks, Hnsw.ef_construction, Hnsw.ef_search));
                default:
                    return std::shared_ptr<BASE>(new BruteForceEuclidean<INDEX_t, DISTANCE_t, QUERY_t>(ndim, nobs, vals, BruteForce.copy));
            }
        } else {
            switch(algorithm) {
                case KMKNN:
                    return std::shared_ptr<BASE>(new KmknnManhattan<INDEX_t, DISTANCE_t, QUERY_t>(ndim, nobs, vals, Kmknn.power));
                case VPTREE:
                    return std::shared_ptr<BASE>(new VpTreeManhattan<INDEX_t, DISTANCE_t, QUERY_t>(ndim, nobs, vals, VpTree.copy));
                case ANNOY:
                    return std::shared_ptr<BASE>(new AnnoyManhattan<INDEX_t, DISTANCE_t, QUERY_t>(ndim, nobs, vals, Annoy.ntrees, Annoy.search_mult));
                case HNSW:
                    return std::shared_ptr<BASE>(new HnswManhattan<INDEX_t, DISTANCE_t, QUERY_t>(ndim, nobs, vals, Hnsw.nlinks, Hnsw.ef_construction, Hnsw.ef_search));
                default:
                    return std::shared_ptr<BASE>(new BruteForceManhattan<INDEX_t, DISTANCE_t, QUERY_t>(ndim, nobs, vals, BruteForce.copy));
            }
        }
    }
};

}

#endif

