#ifndef KNNCOLLE_KMKNN_HPP
#define KNNCOLLE_KMKNN_HPP

#include "distances.hpp"
#include "NeighborQueue.hpp"
#include "Prebuilt.hpp"
#include "Builder.hpp"
#include "MockMatrix.hpp"

#include "kmeans/kmeans.hpp"

#include <algorithm>
#include <vector>
#include <memory>
#include <limits>

/**
 * @file Kmknn.hpp
 *
 * @brief Implements the k-means with k-nearest neighbors (KMKNN) algorithm.
 */

namespace knncolle {

/** 
 * @brief Options for `KmknnBuilder` and `KmknnPrebuilt` construction. 
 * @tparam Store_ Floating point type for the stored data. 
 * For `KmknnBuilder`, this should be the same as `MockMatrix::data_type`.
 * @tparam Index_ Integer type for the indices.
 * For `KmknnBuilder`, this should be the same as `MockMatrix::index_type`.
 * @tparam Dim_ Integer type for the number of dimensions.
 * For `KmknnBuilder`, this should be the same as `MockMatrix::dimension_type`.
 */
template<typename Store_ = double, typename Index_ = int, typename Dim_ = int> 
struct KmknnOptions {
    /**
     * Power of the number of observations, to define the number of cluster centers.
     * By default, a square root is performed.
     */
    double power = 0.5;

    // Note that we use Store_ as the k-means output type, as we'll
    // be storing the cluster centers as Store_'s, not Float_'s.

    /**
     * Initialization method for the k-means clustering.
     * If NULL, defaults to `kmeans::InitializeKmeanspp`.
     */
    std::shared_ptr<kmeans::Initialize<kmeans::SimpleMatrix<Store_, Index_, Dim_>, Index_, Store_> > initialize_algorithm;

    /**
     * Refinement method for the k-means clustering.
     * If NULL, defaults to `kmeans::RefineHartiganWong`.
     */
    std::shared_ptr<kmeans::Refine<kmeans::SimpleMatrix<Store_, Index_, Dim_>, Index_, Store_> > refine_algorithm;
};

/**
 * @brief Index for a KMKNN search.
 *
 * Instances of this class are usually constructed using `KmknnBuilder`.
 *
 * @tparam Distance_ A distance calculation class satisfying the `MockDistance` contract.
 * @tparam Store_ Floating point type for the stored data. 
 * For the output of `KmknnBuilder::build`, this is set to `MockMatrix::data_type`.
 * This may be set to a lower-precision type than `Float_` to save memory.
 * @tparam Dim_ Integer type for the number of dimensions.
 * For the output of `KmknnBuilder::build`, this is set to `MockMatrix::dimension_type`.
 * @tparam Index_ Integer type for the indices.
 * For the output of `KmknnBuilder::build`, this is set to `MockMatrix::index_type`.
 * @tparam Float_ Floating point type for the query data and distances.
 */
template<class Distance_, typename Store_, typename Dim_, typename Index_, typename Float_>
class KmknnPrebuilt : public Prebuilt<Dim_, Index_, Float_> {
private:
    Dim_ my_dim;
    Index_ my_obs;
    size_t my_long_ndim;

public:
    Index_ num_observations() const {
        return my_obs;
    }
    
    Dim_ num_dimensions() const {
        return my_dim;
    }

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
     * @param options Options for constructing the k-means index.
     */
    KmknnPrebuilt(Dim_ num_dim, Index_ num_obs, std::vector<Store_> data, const KmknnOptions<Store_, Index_, Dim_>& options) :
        my_dim(num_dim),
        my_obs(num_obs),
        my_long_ndim(my_dim),
        my_data(std::move(data))
    { 
        auto init = options.initialize_algorithm;
        if (init == nullptr) {
            init.reset(new kmeans::InitializeKmeanspp<kmeans::SimpleMatrix<Store_, Index_, Dim_>, Index_, Store_>);
        }
        auto refine = options.refine_algorithm;
        if (refine == nullptr) {
            refine.reset(new kmeans::RefineHartiganWong<kmeans::SimpleMatrix<Store_, Index_, Dim_>, Index_, Store_>);
        }

        Index_ ncenters = std::ceil(std::pow(my_obs, options.power));
        my_centers.resize(static_cast<size_t>(ncenters) * my_long_ndim); // cast to avoid overflow problems.

        kmeans::SimpleMatrix mat(my_dim, my_obs, my_data.data());
        std::vector<Index_> clusters(my_obs);
        auto output = kmeans::compute(mat, *init, *refine, ncenters, my_centers.data(), clusters.data());

        // Removing empty clusters, e.g., due to duplicate points.
        {
            my_sizes.resize(ncenters);
            std::vector<Index_> remap(ncenters);
            Index_ survivors = 0;
            for (Index_ c = 0; c < ncenters; ++c) {
                if (output.sizes[c]) {
                    if (c > survivors) {
                        auto src = my_centers.begin() + static_cast<size_t>(c) * my_long_ndim; // cast to avoid overflow.
                        auto dest = my_centers.begin() + static_cast<size_t>(survivors) * my_long_ndim;
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
                my_centers.resize(static_cast<size_t>(ncenters) * my_long_ndim);
            }
        }

        my_offsets.resize(ncenters);
        for (Index_ i = 1; i < ncenters; ++i) {
            my_offsets[i] = my_offsets[i - 1] + my_sizes[i - 1];
        }

        // Organize points correctly; firstly, sorting by distance from the assigned center.
        std::vector<std::pair<Float_, Index_> > by_distance(my_obs);
        {
            auto sofar = my_offsets;
            auto host = my_data.data();
            for (Index_ o = 0; o < my_obs; ++o) {
                auto optr = host + static_cast<size_t>(o) * my_long_ndim;
                auto clustid = clusters[o];
                auto cptr = my_centers.data() + static_cast<size_t>(clustid) * my_long_ndim;

                auto& counter = sofar[clustid];
                auto& current = by_distance[counter];
                current.first = Distance_::normalize(Distance_::template raw_distance<Float_>(optr, cptr, my_dim));
                current.second = o;

                ++counter;
            }

            for (Index_ c = 0; c < ncenters; ++c) {
                auto begin = by_distance.begin() + my_offsets[c];
                std::sort(begin, begin + my_sizes[c]);
            }
        }

        // Permuting in-place to mirror the reordered distances, so that the search is more cache-friendly.
        {
            auto host = my_data.data();
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

                // We recursively perform a "thread" of replacements until we
                // are able to find the home of the originally replaced 'o'.
                auto optr = host + static_cast<size_t>(o) * my_long_ndim;
                std::copy_n(optr, my_dim, buffer.begin());
                Index_ replacement = current.second;
                do {
                    auto rptr = host + static_cast<size_t>(replacement) * my_long_ndim;
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
    void search_nn(const Query_* target, internal::NeighborQueue<Index_, Float_>& nearest) const { 
        /* Computing distances to all centers and sorting them. The aim is to
         * go through the nearest centers first, to get the shortest
         * 'threshold' possible through the rest of the search.
         */
        std::vector<std::pair<Float_, Index_> > center_order;
        {
            center_order.reserve(my_sizes.size());
            auto clust_ptr = my_centers.data();
            for (size_t c = 0; c < my_sizes.size(); ++c, clust_ptr += my_dim) {
                center_order.emplace_back(Distance_::template raw_distance<Float_>(target, clust_ptr, my_dim), c);
            }
            std::sort(center_order.begin(), center_order.end());
        }

        // Computing the distance to each center, and deciding whether to proceed for each cluster.
        Float_ threshold_raw = -1;
        for (const auto& curcent : center_order) {
            const Index_ center = curcent.second;
            const Float_ dist2center = Distance_::normalize(curcent.first);

            const auto cur_nobs = my_sizes[center];
            const Float_* dIt = my_dist_to_centroid.data() + my_offsets[center];
            const Float_ maxdist = *(dIt + cur_nobs - 1);

            Index_ firstcell = 0;
#if KNNCOLLE_KMKNN_USE_UPPER
            Float_ upper_bd = std::numeric_limits<Float_>::max();
#endif
            
            if (threshold_raw >= 0) {
                const Float_ threshold = Distance_::normalize(threshold_raw);

                /* The conditional expression below exploits the triangle inequality; it is equivalent to asking whether:
                 *     threshold + maxdist < dist2center
                 * All points (if any) within this cluster with distances above 'lower_bd' are potentially countable.
                 */
                const Float_ lower_bd = dist2center - threshold;
                if (maxdist < lower_bd) {
                    continue;
                }

                firstcell = std::lower_bound(dIt, dIt + cur_nobs, lower_bd) - dIt;

#if KNNCOLLE_KMKNN_USE_UPPER
                /* This exploits the reverse triangle inequality, to ignore points where:
                 *     threshold + dist2center < point-to-center distance
                 */
                upper_bd = threshold + dist2center;
#endif
            }

            const auto cur_start = my_offsets[center];
            const auto* other_cell = my_data.data() + my_long_ndim * static_cast<size_t>(cur_start + firstcell); // cast to avoid overflow issues.
            for (auto celldex = firstcell; celldex < cur_nobs; ++celldex, other_cell += my_dim) {
#if KNNCOLLE_KMKNN_USE_UPPER
                if (*(dIt + celldex) > upper_bd) {
                    break;
                }
#endif

                auto dist2cell_raw = Distance_::template raw_distance<Float_>(target, other_cell, my_dim);
                nearest.add(cur_start + celldex, dist2cell_raw);
                if (nearest.is_full()) {
                    threshold_raw = nearest.limit(); // Shrinking the threshold, if an earlier NN has been found.
#if KNNCOLLE_KMKNN_USE_UPPER
                    upper_bd = Distance_::normalize(threshold_raw) + dist2center; 
#endif
                }
            }
        }
    }

    void normalize(std::vector<std::pair<Index_, Float_> >& output) const {
        for (auto& s : output) {
            s.first = my_observation_id[s.first];
            s.second = Distance_::normalize(s.second);
        }
    }

public:
    void search(Index_ i, Index_ k, std::vector<std::pair<Index_, Float_> >& output) const {
        internal::NeighborQueue<Index_, Float_> nearest(k + 1);
        auto new_i = my_new_location[i];
        auto iptr = my_data.data() + static_cast<size_t>(new_i) * my_long_ndim; // cast to avoid overflow.
        search_nn(iptr, nearest);
        nearest.report(output, new_i);
        normalize(output);
    }

    void search(const Float_* query, Index_ k, std::vector<std::pair<Index_, Float_> >& output) const {
        internal::NeighborQueue<Index_, Float_> nearest(k);
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
 * @tparam Matrix_ Matrix-like object satisfying the `MockMatrix` contract.
 * @tparam Float_ Floating point type for the query data and output distances.
 *
 * @see
 * Wang X (2012). 
 * A fast exact k-nearest neighbors algorithm for high dimensional search using k-means clustering and triangle inequality. 
 * _Proc Int Jt Conf Neural Netw_, 43, 6:2351-2358.
 */
template<class Distance_ = EuclideanDistance, class Matrix_ = SimpleMatrix<double, int, int>, typename Float_ = double>
class KmknnBuilder : public Builder<Matrix_, Float_> {
private:
    KmknnOptions<typename Matrix_::data_type, typename Matrix_::index_type, typename Matrix_::dimension_type> my_options;

public:
    /**
     * @param options Further options for the KMKNN algorithm.
     */
    KmknnBuilder(const KmknnOptions<typename Matrix_::data_type, typename Matrix_::index_type, typename Matrix_::dimension_type>& options) : 
        my_options(std::move(options)) {}

    /**
     * Default constructor.
     */
    KmknnBuilder() = default;

public:
    Prebuilt<typename Matrix_::dimension_type, typename Matrix_::index_type, Float_>* build_raw(const Matrix_& data) const {
        auto ndim = data.num_dimensions();
        auto nobs = data.num_observations();

        typedef typename Matrix_::data_type Store_;
        std::vector<Store_> store(static_cast<size_t>(ndim) * static_cast<size_t>(nobs));

        auto work = data.create_workspace();
        auto sIt = store.begin();
        for (decltype(nobs) o = 0; o < nobs; ++o, sIt += ndim) {
            auto ptr = data.get_observation(work);
            std::copy_n(ptr, ndim, sIt);
        }

        return new KmknnPrebuilt<Distance_, Store_, decltype(ndim), decltype(nobs), Float_>(ndim, nobs, std::move(store), my_options);
    }
};

}

#endif
