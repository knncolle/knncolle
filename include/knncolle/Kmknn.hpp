#ifndef KNNCOLLE_KMKNN_HPP
#define KNNCOLLE_KMKNN_HPP

#include "distances.hpp"
#include "NeighborQueue.hpp"
#include "Prebuilt.hpp"
#include "Builder.hpp"
#include "MockMatrix.hpp"
#include "report_all_neighbors.hpp"

#include "kmeans/kmeans.hpp"

#include <algorithm>
#include <vector>
#include <memory>
#include <limits>
#include <cmath>

/**
 * @file Kmknn.hpp
 *
 * @brief Implements the k-means with k-nearest neighbors (KMKNN) algorithm.
 */

namespace knncolle {

/** 
 * @brief Options for `KmknnBuilder` and `KmknnPrebuilt` construction. 
 *
 * This can also be created via the `KmknnBuilder::Options` definition,
 * which ensures consistency with the template parameters used in `KmknnBuilder`.
 *
 * @tparam Dim_ Integer type for the number of dimensions.
 * When constructing a `KmknnBuilder`, this should be the same as `Matrix_::dimension_type`.
 * @tparam Index_ Integer type for the indices.
 * When constructing a `KmknnBuilder`, this should be the same as `Matrix_::index_type`.
 * @tparam Store_ Floating point type for the stored data. 
 * When constructing a `KmknnBuilder`, this should be the same as `Matrix_::data_type`.
 */
template<typename Dim_ = int, typename Index_ = int, typename Store_ = double>
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


template<class Distance_, typename Dim_, typename Index_, typename Store_, typename Float_>
class KmknnPrebuilt;

/**
 * @brief KMKNN searcher.
 *
 * Instances of this class are usually constructed using `KmknnPrebuilt::initialize()`.
 *
 * @tparam Distance_ A distance calculation class satisfying the `MockDistance` contract.
 * @tparam Dim_ Integer type for the number of dimensions.
 * @tparam Index_ Integer type for the indices.
 * @tparam Store_ Floating point type for the stored data. 
 * @tparam Float_ Floating point type for the query data and output distances.
 */
template<class Distance_, typename Dim_, typename Index_, typename Store_, typename Float_>
class KmknnSearcher : public Searcher<Index_, Float_> {
public:
    /**
     * @cond
     */
    KmknnSearcher(const KmknnPrebuilt<Distance_, Dim_, Index_, Store_, Float_>* parent) : my_parent(parent) {
        center_order.reserve(my_parent->my_sizes.size());
    }
    /**
     * @endcond
     */

private:                
    const KmknnPrebuilt<Distance_, Dim_, Index_, Store_, Float_>* my_parent;
    internal::NeighborQueue<Index_, Float_> my_nearest;
    std::vector<std::pair<Float_, Index_> > my_all_neighbors;
    std::vector<std::pair<Float_, Index_> > center_order;

public:
    void search(Index_ i, Index_ k, std::vector<Index_>* output_indices, std::vector<Float_>* output_distances) {
        my_nearest.reset(k + 1);
        auto new_i = my_parent->my_new_location[i];
        auto iptr = my_parent->my_data.data() + static_cast<size_t>(new_i) * my_parent->my_long_ndim; // cast to avoid overflow.
        my_parent->search_nn(iptr, my_nearest, center_order);
        my_nearest.report(output_indices, output_distances, new_i);
        my_parent->normalize(output_indices, output_distances);
    }

    void search(const Float_* query, Index_ k, std::vector<Index_>* output_indices, std::vector<Float_>* output_distances) {
        if (k == 0) { // protect the NeighborQueue from k = 0.
            internal::flush_output(output_indices, output_distances, 0);
        } else {
            my_nearest.reset(k);
            my_parent->search_nn(query, my_nearest, center_order);
            my_nearest.report(output_indices, output_distances);
            my_parent->normalize(output_indices, output_distances);
        }
    }

    bool can_search_all() const {
        return true;
    }

    Index_ search_all(Index_ i, Float_ d, std::vector<Index_>* output_indices, std::vector<Float_>* output_distances) {
        auto new_i = my_parent->my_new_location[i];
        auto iptr = my_parent->my_data.data() + static_cast<size_t>(new_i) * my_parent->my_long_ndim; // cast to avoid overflow.

        if (!output_indices && !output_distances) {
            Index_ count = 0;
            my_parent->template search_all<true>(iptr, d, count);
            return internal::safe_remove_self(count);

        } else {
            my_all_neighbors.clear();
            my_parent->template search_all<false>(iptr, d, my_all_neighbors);
            internal::report_all_neighbors(my_all_neighbors, output_indices, output_distances, new_i);
            my_parent->normalize(output_indices, output_distances);
            return internal::safe_remove_self(my_all_neighbors.size());
        }
    }

    Index_ search_all(const Float_* query, Float_ d, std::vector<Index_>* output_indices, std::vector<Float_>* output_distances) {
        if (!output_indices && !output_distances) {
            Index_ count = 0;
            my_parent->template search_all<true>(query, d, count);
            return count;

        } else {
            my_all_neighbors.clear();
            my_parent->template search_all<false>(query, d, my_all_neighbors);
            internal::report_all_neighbors(my_all_neighbors, output_indices, output_distances);
            my_parent->normalize(output_indices, output_distances);
            return my_all_neighbors.size();
        }
    }
};

/**
 * @brief Index for a KMKNN search.
 *
 * Instances of this class are usually constructed using `KmknnBuilder::build_raw()`.
 *
 * @tparam Distance_ A distance calculation class satisfying the `MockDistance` contract.
 * @tparam Dim_ Integer type for the number of dimensions.
 * For the output of `KmknnBuilder::build_raw()`, this is set to `Matrix_::dimension_type`.
 * @tparam Index_ Integer type for the indices.
 * For the output of `KmknnBuilder::build_raw()`, this is set to `Matrix_::index_type`.
 * @tparam Store_ Floating point type for the stored data. 
 * For the output of `KmknnBuilder::build_raw()`, this is set to `Matrix_::data_type`.
 * This may be set to a lower-precision type than `Float_` to save memory.
 * @tparam Float_ Floating point type for the query data and distances.
 */
template<class Distance_, typename Dim_, typename Index_, typename Store_, typename Float_>
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
    KmknnPrebuilt(Dim_ num_dim, Index_ num_obs, std::vector<Store_> data, const KmknnOptions<Dim_, Index_, Store_>& options) :
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
                my_sizes.resize(ncenters);
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
    void search_nn(const Query_* target, internal::NeighborQueue<Index_, Float_>& nearest, std::vector<std::pair<Float_, Index_> >& center_order) const { 
        /* Computing distances to all centers and sorting them. The aim is to
         * go through the nearest centers first, to try to get the shortest
         * threshold (i.e., 'nearest.limit()') possible at the start;
         * this allows us to skip searches of the later clusters.
         */
        center_order.clear();
        size_t ncenters = my_sizes.size();
        center_order.reserve(ncenters);
        auto clust_ptr = my_centers.data();
        for (size_t c = 0; c < ncenters; ++c, clust_ptr += my_dim) {
            center_order.emplace_back(Distance_::template raw_distance<Float_>(target, clust_ptr, my_dim), c);
        }
        std::sort(center_order.begin(), center_order.end());

        // Computing the distance to each center, and deciding whether to proceed for each cluster.
        Float_ threshold_raw = std::numeric_limits<Float_>::infinity();
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

            if (!std::isinf(threshold_raw)) {
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
                if (dist2cell_raw <= threshold_raw) {
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
    }

    template<bool count_only_, typename Query_, typename Output_>
    void search_all(const Query_* target, Float_ threshold, Output_& all_neighbors) const {
        Float_ threshold_raw = Distance_::denormalize(threshold);

        /* Computing distances to all centers. We don't sort them here 
         * because the threshold is constant so there's no point.
         */
        Index_ ncenters = my_sizes.size();
        auto center_ptr = my_centers.data(); 
        for (Index_ center = 0; center < ncenters; ++center, center_ptr += my_dim) {
            const Float_ dist2center = Distance_::normalize(Distance_::template raw_distance<Float_>(target, center_ptr, my_dim));

            auto cur_nobs = my_sizes[center];
            const Float_* dIt = my_dist_to_centroid.data() + my_offsets[center];
            const Float_ maxdist = *(dIt + cur_nobs - 1);

            /* The conditional expression below exploits the triangle inequality; it is equivalent to asking whether:
             *     threshold + maxdist < dist2center
             * All points (if any) within this cluster with distances above 'lower_bd' are potentially countable.
             */
            const Float_ lower_bd = dist2center - threshold;
            if (maxdist < lower_bd) {
                continue;
            }

            Index_ firstcell = std::lower_bound(dIt, dIt + cur_nobs, lower_bd) - dIt;
#if KNNCOLLE_KMKNN_USE_UPPER
            /* This exploits the reverse triangle inequality, to ignore points where:
             *     threshold + dist2center < point-to-center distance
             */
            Float_ upper_bd = threshold + dist2center;
#endif

            const auto cur_start = my_offsets[center];
            auto other_ptr = my_data.data() + my_long_ndim * static_cast<size_t>(cur_start + firstcell); // cast to avoid overflow issues.
            for (auto celldex = firstcell; celldex < cur_nobs; ++celldex, other_ptr += my_dim) {
#if KNNCOLLE_KMKNN_USE_UPPER
                if (*(dIt + celldex) > upper_bd) {
                    break;
                }
#endif

                auto dist2cell_raw = Distance_::template raw_distance<Float_>(target, other_ptr, my_dim);
                if (dist2cell_raw <= threshold_raw) {
                    if constexpr(count_only_) {
                        ++all_neighbors;
                    } else {
                        all_neighbors.emplace_back(dist2cell_raw, cur_start + celldex);
                    }
                }
            }
        }
    }

    void normalize(std::vector<Index_>* output_indices, std::vector<Float_>* output_distances) const {
        if (output_indices) {
            for (auto& s : *output_indices) {
                s = my_observation_id[s];
            }
        }
        if (output_distances) {
            for (auto& d : *output_distances) {
                d = Distance_::normalize(d);
            }
        }
    }

    friend class KmknnSearcher<Distance_, Dim_, Index_, Store_, Float_>;

public:
    /**
     * Creates a `KmknnSearcher` instance.
     */
    std::unique_ptr<Searcher<Index_, Float_> > initialize() const {
        return std::make_unique<KmknnSearcher<Distance_, Dim_, Index_, Store_, Float_> >(this);
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
template<class Distance_ = EuclideanDistance, class Matrix_ = SimpleMatrix<int, int, double>, typename Float_ = double>
class KmknnBuilder : public Builder<Matrix_, Float_> {
public:
    /**
     * Convenient name for the `KmknnOptions` class that ensures consistent template parametrization.
     */
    typedef KmknnOptions<typename Matrix_::dimension_type, typename Matrix_::index_type, typename Matrix_::data_type> Options;

private:
    Options my_options;

public:
    /**
     * @param options Further options for the KMKNN algorithm.
     */
    KmknnBuilder(Options options) : my_options(std::move(options)) {}

    /**
     * Default constructor.
     */
    KmknnBuilder() = default;

    /**
     * @return Options for the KMKNN algorithm.
     * These can be modified prior to running `build_raw()` and friends.
     */
    Options& get_options() {
        return my_options;
    }

public:
    /**
     * Creates a `KmknnPrebuilt` instance.
     */
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

        return new KmknnPrebuilt<Distance_, decltype(ndim), decltype(nobs), Store_, Float_>(ndim, nobs, std::move(store), my_options);
    }
};

}

#endif
