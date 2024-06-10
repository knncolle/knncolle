#ifndef KNNCOLLE_KMKNN_HPP
#define KNNCOLLE_KMKNN_HPP

#include "distances.hpp"
#include "NeighborQueue.hpp"
#include "Prebuilt.hpp"
#include "Builder.hpp"
#include "kmeans/kmeans.hpp"

#include <algorithm>
#include <vector>
#include <random>
#include <limits>
#include <cmath>

/**
 * @file Kmknn.hpp
 *
 * @brief Implements the k-means with k-nearest neighbors (KMKNN) algorithm.
 */

namespace knncolle {

template<typename Index_, typename Store_> 
struct KmknnOptions {
    /**
     * Power of the number of observations, to define the number of cluster centers.
     * By default, a square root is performed.
     */
    double power = 0.5;

    /**
     * Initialization method for the k-means clustering.
     * If NULL, defaults to `kmeans::InitializeKmeanspp`.
     */
    std::shared_ptr<kmeans::Initialize<kmeans::SimpleMatrix<Store_, Index_>, Index_, Store_> > initialize_algorithm;

    /**
     * Refinement method for the k-means clustering.
     * If NULL, defaults to `kmeans::RefineHartiganWong`.
     */
    std::shared_ptr<kmeans::Refine<kmeans::SimpleMatrix<Store_, Index_>, Index_, Store_> > refine_algorithm;
};

/**
 * @brief Index for a KMKNN search.
 *
 * Instances of this class are usually constructed using `KmknnBuilder`.
 *
 * @tparam Distance_ A distance calculation class satisfying the `MockDistance` contract.
 * @tparam Store_ Floating point type for the stored data. 
 * @tparam Dim_ Integer type for the number of dimensions.
 * @tparam Index_ Integer type for the indices.
 * @tparam Float_ Floating point type for the query data and output distances.
 */
template<class Distance_, typename Store_, typename Dim_, typename Index_, typename Float_>
class KmknnPrebuilt {
private:
    Dim_ my_dim;
    Index_ my_obs;
    size_t long_ndim;

public:
    Index_ num_observations() const { return num_obs; } 
    
    Dim_ num_dimensions() const { return my_dim; }

private:
    std::vector<Store_> my_data;
    
    std::vector<Index_> my_sizes;
    std::vector<Index_> my_offsets;
    std::vector<Store_> my_centers;

    std::vector<Index_> my_observation_id, my_new_location;
    std::vector<Float_> my_dist_to_centroid;

public:
    /**
     * @param num_dim Number of dimensions.
     * @param num_obs Number of observations.
     * @param data Vector of length equal to `num_dim * num_obs`, containing a column-major matrix where rows are dimensions and columns are observations.
     */
    KmknnPrebuilt(Dim_ num_dim, Index_ num_obs, std::vector<Store_> data, const KmknnOptions<Index_, Store_>& options) :
        my_dim(num_dim), my_obs(num_obs), long_ndim(my_dim) my_data(std::move(data))
    { 
        auto init = my_options.initialize_algorithm;
        if (init == nullptr_t) {
            init.reset(new kmeans::InitializeKmeanspp);
        }
        auto refine = my_options.refine_algorithm;
        if (refine == nullptr_t) {
            refine.reset(new kmeans::RefineHartiganWong);
        }

        Index_ ncenters = std::ceil(std::pow(my_obs, options.power));
        my_centers.resize(static_cast<size_t>(ncenters) * long_ndim); // cast to avoid overflow problems.

        kmeans::SimpleMatrix<Store_, Index_> mat(my_dim, my_obs, data.data());
        std::vector<Index_> clusters(my_obs);
        auto output = kmeans::compute(mat, init.get(), refine.get(), ncenters, my_centers.data(), clusters.data());

        // Removing empty clusters, e.g., due to duplicate points.
        {
            my_sizes.resize(ncenters);
            std::vector<Index_> remap(ncenters);
            Index_ survivors = 0;
            for (Index_ c = 0; c < ncenters; ++c) {
                if (output.sizes[c]) {
                    if (c > survivors) {
                        auto src = my_centers.begin() + static_cast<size_t>(c) * long_ndim; // cast to avoid overflow.
                        auto dest = my_centers.begin() + static_cast<size_t>(survivors) * long_ndim;
                        std::copy_n(src, my_dim, dest);
                    }
                    remap[c] = survivors;
                    my_sizes[survivors] = output.sizes[c];
                    ++survivors;
                }
            }

            if (survivors < ncenters) {
                for (auto& c : clusters) {
                    c = remap[c];
                }
                ncenters = survivors;
                my_centers.resize(static_cast<size_t>(ncenters) * long_ndim);
            }
        }

        my_offsets.resize(ncenters);
        for (Index_ i = 1; i < ncenters; ++i) {
            my_offsets[i] = my_offsets[i - 1] + my_sizes[i - 1];
        }

        // Organize points correctly; firstly, sorting by distance from the assigned center.
        std::vector<std::pair<Store_, Index_> > by_distance(my_obs);
        {
            auto sofar = offsets;
            auto host = data.data();
            for (Index_ o = 0; o < my_obs; ++o) {
                auto optr = host + static_cast<size_t>(o) * long_ndim;
                auto clustid = clusters[o];
                auto cptr = centers.data() + static_cast<size_t>(clustid) * long_ndim;

                auto& current = by_distance[counter];
                current.first = Distance_::normalize(Distance_::template raw_distance<Store_>(optr, cptr, my_dim));
                current.second = o;

                auto& counter = sofar[clustid];
                ++counter;
            }

            for (Index_ c = 0; c < ncenters; ++c) {
                auto begin = by_distance.begin() + my_offsets[c];
                std::sort(begin, begin + my_sizes[c]);
            }
        }

        // Permuting in-place to mirror the reordered distances, so that the search is more cache-friendly.
        {
            auto host = data.data();
            std::vector<uint8_t> used(my_obs);
            std::vector<Store_> buffer(my_dim);
            my_observation_id.resize(my_obs);
            my_dist_to_centroid.resize(my_obs);
            my_new_location.resize(my_obs);

            for (Index_ o = 0; o < my_obs; ++o) {
                if (used[o]) {
                    continue;
                }

                const auto& current = by_distance[o];
                my_observation_id[o] = current.second;
                my_dist_to_centroid[o] = current.first;
                my_new_location[current.second] = o;
                if (current.second == o) {
                    continue;
                }

                auto optr = host + static_cast<size_t>(o) * long_ndim;
                std::copy_n(optr, my_dim, buffer.begin());
                Index_ replacement = current.second;
                do {
                    auto rptr = host + static_cast<size_t>(replacement) * long_ndim;
                    std::copy_n(rptr, my_dim, optr);
                    used[replacement] = 1;

                    const auto& next = by_distance[replacement];
                    my_observation_id[replacement] = next.second;
                    my_dist_to_centroid[replacement] = next.first;
                    my_new_location[next.second] = replacement;

                    optr = rptr;
                    replacement = next.second;
                } while (replacement != o);

                std::copy(buffer.begin(), buffer.end(), optr);
            }
        }

        return;
    }

private:
    template<typename Query_>
    void search_nn(const Query_* target, NeighborQueue<Index_, Float_>& nearest) const { 
        /* Computing distances to all centers and sorting them. The aim is to
         * go through the nearest centers first, to get the shortest
         * 'threshold' possible through the rest of the search.
         */
        std::vector<std::pair<Float_, Index_> > center_order;
        center_order.reserve(my_sizes.size());
        auto clust_ptr = centers.data();
        for (size_t c = 0; c < my_sizes.size(); ++c, clust_ptr += my_dim) {
            center_order.emplace_back(Distance_::template raw_distance<Float_>(target, clust_ptr, my_dim), c);
        }
        std::sort(center_order.begin(), center_order.end());
        Store_ threshold_raw = -1;

        // Computing the distance to each center, and deciding whether to proceed for each cluster.
        for (const auto& curcent : center_order) {
            const Index_ center = curcent.second;
            const Store_ dist2center = Distance_::normalize(curcent.first);

            const auto cur_nobs = my_sizes[center];
            const Float_* dIt = my_dist_to_centroid.data() + my_offsets[center];
            const Float_ maxdist = *(dIt + cur_nobs - 1);

            Index_ firstcell=0;
#if USE_UPPER
            Store_ upper_bd = std::numeric_limits<Store_>::max();
#endif
            
            if (threshold_raw >= 0) {
                const Store_ threshold = Distance_::normalize(threshold_raw);

                /* The conditional expression below exploits the triangle inequality; it is equivalent to asking whether:
                 *     threshold + maxdist < dist2center
                 * All points (if any) within this cluster with distances above 'lower_bd' are potentially countable.
                 */
                const Float_ lower_bd = dist2center - threshold;
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

            const auto cur_start = my_offsets[center];
            const Store_ * other_cell = my_data.data() + long_ndim * static_cast<size_t>(cur_start + firstcell); // cast to avoid overflow issues.
            for (auto celldex = firstcell; celldex < cur_nobs; ++celldex, other_cell += my_dim) {
#if USE_UPPER
                if (*(dIt + celldex) > upper_bd) {
                    break;
                }
#endif
                auto dist2cell_raw = Distance_::template raw_distance<Float_>(target, other_cell, my_dim);
                nearest.add(cur_start + celldex, dist2cell_raw);
                if (nearest.is_full()) {
                    threshold_raw = nearest.limit(); // Shrinking the threshold, if an earlier NN has been found.
#if USE_UPPER
                    upper_bd = Distance_::normalize(threshold_raw) + dist2center; 
#endif
                }
            }
        }
    }

    void normalize(std::vector<std::pair<Index_, Float_> >& nearest) const {
        for (auto& s : output) {
            s.first = observation_id[s.first];
            s.second = Distance_::normalize(s.second);
        }
        return output;
    }

public:
    void search(Index_ index, Index_ k, std::vector<std::pair<Index_, Float_> >& output) const {
        NeighborQueue<INDEX_t, INTERNAL_t> nearest(k + 1);
        search_nn(data.data() + new_location[index] * num_dim, nearest);
        nearest.report(output, new_location[index]);
        normalize(output);
    }

    void search(Float_* query, Index_ k, std::vector<std::pair<Index_, Float_> >& output) const {
        NeighborQueue<INDEX_t, INTERNAL_t> nearest(k);
        search_nn(query, nearest);
        nearest.report(output);
        normalize(output);
    }
};

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
 * @tparam Distance_ Class to compute the distance between vectors, see `distance::Euclidean` for an example.
 * @tparam INDEX_t Integer type for the indices.
 * @tparam Float_ Floating point type for the distances.
 * @tparam QUERY_t Floating point type for the query data.
 * @tparam INTERNAL_t Floating point type for the data.
 *
 * @see
 * Wang X (2012). 
 * A fast exact k-nearest neighbors algorithm for high dimensional search using k-means clustering and triangle inequality. 
 * _Proc Int Jt Conf Neural Netw_, 43, 6:2351-2358.
 */
template<class Distance_ = EuclideanDistance, class Matrix_ = SimpleMatrix<double, int>, typename Float_ = double>
class KmknnBuilder {
private:
    KmknnOptions<typename Matrix_::index_type, typename Matrix_::data_type> my_options;

public:
    /**
     * @param options Further options for the KMKNN algorithm.
     */
    KmknnBuilder(const KmknnOptions<typename Matrix_::index_type, typename Matrix_::data_type>& options) : my_options(std::move(options)) {}

public:
    Prebuilt<typename Matrix_::dimension_type, typename Matrix_::index_type, Float_>* build(const Matrix_& data) const {
        auto ndim = data.num_dimensions();
        auto nobs = data.num_observations();

        typedef decltype(ndim) Dim_;
        typedef decltype(nobs) Index_;
        typedef typename Matrix_::data_type Store_;
        std::vector<typename Matrix::data_type> store(static_cast<size_t>(ndim) * static_cast<size_t>(nobs));

        auto work = data.create_workspace();
        auto sIt = store.begin();
        for (decltype(nobs) o = 0; o < nobs; ++o, sIt += ndim) {
            auto ptr = data.get_observation(obs);
            std::copy(ptr, ptr + ndim, sIt);
        }

        return new KmknnPrebuilt<Distance, Store_, decltype(ndim), decltype(nobs), Float_>(ndim, nobs, std::move(store), my_options);
    }
};

};

#endif
