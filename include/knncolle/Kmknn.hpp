#ifndef KNNCOLLE_KMKNN_HPP
#define KNNCOLLE_KMKNN_HPP

#include "../utils/distances.hpp"
#include "../utils/NeighborQueue.hpp"
#include "../utils/Base.hpp"
#include "kmeans/Kmeans.hpp"

#include <algorithm>
#include <vector>
#include <random>
#include <limits>
#include <cmath>

#ifdef DEBUG
#include <iostream>
#endif

#ifndef KMEANS_CUSTOM_PARALLEL
#ifdef KNNCOLLE_CUSTOM_PARALLEL
#define KMEANS_CUSTOM_PARALLEL KNNCOLLE_CUSTOM_PARALLEL
#endif
#endif

/**
 * @file Kmknn.hpp
 *
 * @brief Implements the k-means with k-nearest neighbors (KMKNN) algorithm.
 */

namespace knncolle {

/**
 * @brief Perform a nearest neighbor search based on k-means clustering.
 *
 * In the k-means with k-nearest neighbors (KMKNN) algorithm (Wang, 2012), k-means clustering is first applied to the data points,
 * with the number of cluster centers defined as the square root of the number of points.
 * The cluster assignment and distance to the assigned cluster center for each point represent the KMKNN indexing information, 
 * allowing us to speed up the nearest neighbor search by exploiting the triangle inequality between cluster centers, the query point and each point in the cluster to narrow the search space.
 * The advantage of the KMKNN approach is its simplicity and minimal overhead,
 * resulting in performance improvements over conventional tree-based methods for high-dimensional data where most points need to be searched anyway.
 *
 * @tparam DISTANCE Class to compute the distance between vectors, see `distance::Euclidean` for an example.
 * @tparam INDEX_t Integer type for the indices.
 * @tparam DISTANCE_t Floating point type for the distances.
 * @tparam QUERY_t Floating point type for the query data.
 * @tparam INTERNAL_t Floating point type for the data.
 *
 * @see
 * Wang X (2012). 
 * A fast exact k-nearest neighbors algorithm for high dimensional search using k-means clustering and triangle inequality. 
 * _Proc Int Jt Conf Neural Netw_, 43, 6:2351-2358.
 */
template<class DISTANCE, typename INDEX_t = int, typename DISTANCE_t = double, typename QUERY_t = DISTANCE_t, typename INTERNAL_t = DISTANCE_t>
class Kmknn : public Base<INDEX_t, DISTANCE_t, QUERY_t> {
private:
    INDEX_t num_dim;
    INDEX_t num_obs;

public:
    INDEX_t nobs() const { return num_obs; } 
    
    INDEX_t ndim() const { return num_dim; }

private:
    std::vector<INTERNAL_t> data;
    
    std::vector<INDEX_t> sizes;
    std::vector<INDEX_t> offsets;

    std::vector<INTERNAL_t> centers;

    std::vector<INDEX_t> observation_id, new_location;
    std::vector<DISTANCE_t> dist_to_centroid;

public:
    /**
     * @param ndim Number of dimensions.
     * @param nobs Number of observations.
     * @param vals Pointer to an array of length `ndim * nobs`, corresponding to a dimension-by-observation matrix in column-major format, 
     * i.e., contiguous elements belong to the same observation.
     * @param power Power of `nobs` to define the number of cluster centers.
     * By default, a square root is performed.
     * @param nthreads Number of threads to use for the k-means clustering.
     *
     * @tparam INPUT_t Floating-point type of the input data.
     */
    template<typename INPUT_t>
    Kmknn(INDEX_t ndim, INDEX_t nobs, const INPUT_t* vals, double power = 0.5, int nthreads = 1) : 
            num_dim(ndim), 
            num_obs(nobs), 
            data(ndim * nobs), 
            sizes(std::ceil(std::pow(num_obs, power))), 
            offsets(sizes.size()),
            centers(sizes.size() * ndim),
            observation_id(nobs),
            new_location(nobs),
            dist_to_centroid(nobs)
    { 
        std::vector<int> clusters(num_obs);
        auto ncenters = sizes.size();

        // Try to avoid a copy if we're dealing with the same type;
        // otherwise, we just dump it into 'data', given that we 
        // won't be rewriting it for a while anyway.
        const INTERNAL_t* host;
        if constexpr(std::is_same<INPUT_t, INTERNAL_t>::value) {
            host = vals;
        } else {
            std::copy(vals, vals + data.size(), data.data());
            host = data.data();
        }

        kmeans::Kmeans<INTERNAL_t, int> krunner;
        krunner.set_num_threads(nthreads);
        auto output = kmeans::Kmeans<INTERNAL_t, int>().run(ndim, nobs, host, ncenters, centers.data(), clusters.data());
        std::swap(sizes, output.sizes);

        // In case there were some duplicate points, we just resize this a bit.
        if (ncenters != sizes.size()) {
            ncenters = sizes.size();
            offsets.resize(ncenters);
            centers.resize(ncenters * ndim);
        }

        for (INDEX_t i = 1; i < ncenters; ++i) {
            offsets[i] = offsets[i - 1] + sizes[i - 1];
        }

        // Organize points correctly; firstly, sorting by distance from the assigned center.
        std::vector<std::pair<INTERNAL_t, INDEX_t> > by_distance(nobs);
        {
            auto sofar = offsets;
            for (INDEX_t o = 0; o < nobs; ++o) {
                const auto& clustid = clusters[o];
                auto& counter = sofar[clustid];
                auto& current = by_distance[counter];
                current.first = DISTANCE::normalize(DISTANCE::template raw_distance<INTERNAL_t>(host + o * num_dim, centers.data() + clustid * num_dim, num_dim));
                current.second = o;
                ++counter;
            }

            for (INDEX_t c = 0; c < ncenters; ++c) {
                auto begin = by_distance.begin() + offsets[c];
                std::sort(begin, begin + sizes[c]);
            }
        }

        // Now, copying this over. 
        {
            auto store = data.data();
            for (INDEX_t o = 0; o < nobs; ++o, store += num_dim) {
                const auto& current = by_distance[o];
                auto source = vals + ndim * current.second; // must use 'vals' here, as 'host' might alias 'data'!
                std::copy(source, source + ndim, store);
                observation_id[o] = current.second;
                new_location[current.second] = o;
                dist_to_centroid[o] = current.first;
            }
        }

        return;
    }

    std::vector<std::pair<INDEX_t, DISTANCE_t> > find_nearest_neighbors(INDEX_t index, int k) const {
        NeighborQueue<INDEX_t, INTERNAL_t> nearest(k, new_location[index]);
        search_nn(data.data() + new_location[index] * num_dim, nearest);
        return report(nearest);
    }

    std::vector<std::pair<INDEX_t, DISTANCE_t> > find_nearest_neighbors(const QUERY_t* query, int k) const {
        NeighborQueue<INDEX_t, INTERNAL_t> nearest(k);
        search_nn(query, nearest);
        return report(nearest);
    }

    const QUERY_t* observation(INDEX_t index, QUERY_t* buffer) const {
        auto candidate = data.data() + num_dim * new_location[index];
        if constexpr(std::is_same<QUERY_t, INTERNAL_t>::value) {
            return candidate;
        } else {
            std::copy(candidate, candidate + num_dim, buffer);
            return buffer;
        }
    }

    using Base<INDEX_t, DISTANCE_t, QUERY_t>::observation;

private:
    template<typename INPUT_t>
    void search_nn(INPUT_t* target, NeighborQueue<INDEX_t, INTERNAL_t>& nearest) const { 
        /* Computing distances to all centers and sorting them. The aim is to
         * go through the nearest centers first, to get the shortest
         * 'threshold' possible through the rest of the search.
         */
        std::vector<std::pair<INTERNAL_t, INDEX_t> > center_order(sizes.size());
        auto clust_ptr = centers.data();
        for (size_t c = 0; c < sizes.size(); ++c, clust_ptr += num_dim) {
            center_order[c].first = DISTANCE::template raw_distance<INTERNAL_t>(target, clust_ptr, num_dim);
            center_order[c].second = c;
        }
        std::sort(center_order.begin(), center_order.end());
        INTERNAL_t threshold_raw = -1;

        // Computing the distance to each center, and deciding whether to proceed for each cluster.
        for (const auto& curcent : center_order) {
            const INDEX_t center = curcent.second;
            const INTERNAL_t dist2center = DISTANCE::normalize(curcent.first);

            const auto cur_nobs = sizes[center];
            const DISTANCE_t* dIt = dist_to_centroid.data() + offsets[center];
            const DISTANCE_t maxdist = *(dIt + cur_nobs - 1);

            INDEX_t firstcell=0;
#if USE_UPPER
            INTERNAL_t upper_bd = std::numeric_limits<INTERNAL_t>::max();
#endif
            
            if (threshold_raw >= 0) {
                const INTERNAL_t threshold = DISTANCE::normalize(threshold_raw);

                /* The conditional expression below exploits the triangle inequality; it is equivalent to asking whether:
                 *     threshold + maxdist < dist2center
                 * All points (if any) within this cluster with distances above 'lower_bd' are potentially countable.
                 */
                const DISTANCE_t lower_bd = dist2center - threshold;
                if (maxdist < lower_bd) {
                    continue;
                }
                firstcell=std::lower_bound(dIt, dIt + cur_nobs, lower_bd) - dIt;
#if USE_UPPER
                /* This exploits the reverse triangle inequality, to ignore points where:
                 *     threshold + dist2center < point-to-center distance
                 */
                upper_bd = threshold + dist2center;
#endif
            }

            const auto cur_start = offsets[center];
            const INTERNAL_t * other_cell = data.data() + num_dim * (cur_start + firstcell);
            for (auto celldex = firstcell; celldex < cur_nobs; ++celldex, other_cell += num_dim) {
#if USE_UPPER
                if (*(dIt + celldex) > upper_bd) {
                    break;
                }
#endif
                const auto dist2cell_raw = DISTANCE::template raw_distance<INTERNAL_t>(target, other_cell, num_dim);
                nearest.add(cur_start + celldex, dist2cell_raw);
                if (nearest.is_full()) {
                    threshold_raw = nearest.limit(); // Shrinking the threshold, if an earlier NN has been found.
#if USE_UPPER
                    upper_bd = DISTANCE::normalize(threshold_raw) + dist2center; 
#endif
                }
            }
        }
    }

    template<class QUEUE>
    auto report(QUEUE& nearest) const {
        auto output = nearest.template report<DISTANCE_t>();
        for (auto& s : output) {
            s.first = observation_id[s.first];
            s.second = DISTANCE::normalize(s.second);
        }
        return output;
    }

#ifdef DEBUG
    template<class V>
    void print_vector(const V& input, const char* msg) const {
        std::cout << msg << ": ";
        for (auto v : input) {
            std::cout << v << " ";
        }
        std::cout << std::endl;
    }
#endif
};

/**
 * Perform a KMKNN search with Euclidean distances.
 */
template<typename INDEX_t = int, typename DISTANCE_t = double, typename QUERY_t = DISTANCE_t, typename INTERNAL_t = DISTANCE_t>
using KmknnEuclidean = Kmknn<distances::Euclidean, INDEX_t, DISTANCE_t, QUERY_t, INTERNAL_t>;

/**
 * Perform a KMKNN search with Manhattan distances.
 * Note that k-means clustering may not provide a particularly good indexing structure for Manhattan distances, so your mileage may vary.
 */
template<typename INDEX_t = int, typename DISTANCE_t = double, typename QUERY_t = DISTANCE_t, typename INTERNAL_t = DISTANCE_t>
using KmknnManhattan = Kmknn<distances::Manhattan, INDEX_t, DISTANCE_t, QUERY_t, INTERNAL_t>;

};

#endif
