#ifndef KNNCOLLE_BRUTEFORCE_HPP
#define KNNCOLLE_BRUTEFORCE_HPP

#include "distances.hpp"
#include "NeighborQueue.hpp"
#include "Searcher.hpp"
#include "Builder.hpp"
#include "Prebuilt.hpp"
#include "Matrix.hpp"
#include "report_all_neighbors.hpp"
#include "utils.hpp"

#include <vector>
#include <limits>
#include <memory>
#include <cstddef>
#include <string>
#include <cstring>
#include <filesystem>

#include "sanisizer/sanisizer.hpp"

/**
 * @file Bruteforce.hpp
 *
 * @brief Implements a brute-force search for nearest neighbors.
 */

namespace knncolle {

/**
 * Name of the brute-force algorithm when registering a loading function to `load_prebuilt_registry()`.
 */
inline static constexpr const char* bruteforce_prebuilt_save_name = "knncolle::Bruteforce";

/**
 * @cond
 */
template<typename Index_, typename Data_, typename Distance_, typename DistanceMetric_>
class BruteforcePrebuilt;

template<typename Index_, typename Data_, typename Distance_, class DistanceMetric_>
class BruteforceSearcher final : public Searcher<Index_, Data_, Distance_> {
public:
    BruteforceSearcher(const BruteforcePrebuilt<Index_, Data_, Distance_, DistanceMetric_>& parent) : my_parent(parent) {}

private:                
    const BruteforcePrebuilt<Index_, Data_, Distance_, DistanceMetric_>& my_parent;
    NeighborQueue<Index_, Distance_> my_nearest;
    std::vector<std::pair<Distance_, Index_> > my_all_neighbors;

private:
    void normalize(std::vector<Distance_>* output_distances) const {
        if (output_distances) {
            for (auto& d : *output_distances) {
                d = my_parent.my_metric->normalize(d);
            }
        } 
    }

public:
    void search(Index_ i, Index_ k, std::vector<Index_>* output_indices, std::vector<Distance_>* output_distances) {
        my_nearest.reset(k + 1); // +1 is safe as k < num_obs.
        auto ptr = my_parent.my_data.data() + sanisizer::product_unsafe<std::size_t>(i, my_parent.my_dim);
        my_parent.search(ptr, my_nearest);
        my_nearest.report(output_indices, output_distances, i);
        normalize(output_distances);
    }

    void search(const Data_* query, Index_ k, std::vector<Index_>* output_indices, std::vector<Distance_>* output_distances) {
        if (k == 0) { // protect the NeighborQueue from k = 0.
            if (output_indices) {
                output_indices->clear();
            }
            if (output_distances) {
                output_distances->clear();
            }
        } else {
            my_nearest.reset(k);
            my_parent.search(query, my_nearest);
            my_nearest.report(output_indices, output_distances);
            normalize(output_distances);
        }
    }

    bool can_search_all() const {
        return true;
    }

    Index_ search_all(Index_ i, Distance_ d, std::vector<Index_>* output_indices, std::vector<Distance_>* output_distances) {
        auto ptr = my_parent.my_data.data() + sanisizer::product_unsafe<std::size_t>(i, my_parent.my_dim);

        if (!output_indices && !output_distances) {
            Index_ count = 0;
            my_parent.template search_all<true>(ptr, d, count);
            return count_all_neighbors_without_self(count);

        } else {
            my_all_neighbors.clear();
            my_parent.template search_all<false>(ptr, d, my_all_neighbors);
            report_all_neighbors(my_all_neighbors, output_indices, output_distances, i);
            normalize(output_distances);
            return count_all_neighbors_without_self(my_all_neighbors.size());
        }
    }

    Index_ search_all(const Data_* query, Distance_ d, std::vector<Index_>* output_indices, std::vector<Distance_>* output_distances) {
        if (!output_indices && !output_distances) {
            Index_ count = 0;
            my_parent.template search_all<true>(query, d, count);
            return count;

        } else {
            my_all_neighbors.clear();
            my_parent.template search_all<false>(query, d, my_all_neighbors);
            report_all_neighbors(my_all_neighbors, output_indices, output_distances);
            normalize(output_distances);
            return my_all_neighbors.size();
        }
    }
};

template<typename Index_, typename Data_, typename Distance_, class DistanceMetric_>
class BruteforcePrebuilt final : public Prebuilt<Index_, Data_, Distance_> {
private:
    std::size_t my_dim;
    Index_ my_obs;
    std::vector<Data_> my_data;
    std::shared_ptr<const DistanceMetric_> my_metric;

public:
    BruteforcePrebuilt(std::size_t num_dim, Index_ num_obs, std::vector<Data_> data, std::shared_ptr<const DistanceMetric_> metric) : 
        my_dim(num_dim), my_obs(num_obs), my_data(std::move(data)), my_metric(std::move(metric)) {}

public:
    std::size_t num_dimensions() const {
        return my_dim;
    }

    Index_ num_observations() const {
        return my_obs;
    }

private:
    void search(const Data_* query, NeighborQueue<Index_, Distance_>& nearest) const {
        Distance_ threshold_raw = std::numeric_limits<Distance_>::infinity();
        for (Index_ x = 0; x < my_obs; ++x) {
            auto dist_raw = my_metric->raw(my_dim, query, my_data.data() + sanisizer::product_unsafe<std::size_t>(x, my_dim));
            if (dist_raw <= threshold_raw) {
                nearest.add(x, dist_raw);
                if (nearest.is_full()) {
                    threshold_raw = nearest.limit();
                }
            }
        }
    }

    template<bool count_only_, typename Output_>
    void search_all(const Data_* query, Distance_ threshold, Output_& all_neighbors) const {
        Distance_ threshold_raw = my_metric->denormalize(threshold);
        for (Index_ x = 0; x < my_obs; ++x) {
            Distance_ raw = my_metric->raw(my_dim, query, my_data.data() + sanisizer::product_unsafe<std::size_t>(x, my_dim));
            if (threshold_raw >= raw) {
                if constexpr(count_only_) {
                    ++all_neighbors; // expect this to be an integer.
                } else {
                    all_neighbors.emplace_back(raw, x); // expect this to be a vector of (distance, index) pairs.
                }
            }
        }
    }

    friend class BruteforceSearcher<Index_, Data_, Distance_, DistanceMetric_>;

public:
    std::unique_ptr<Searcher<Index_, Data_, Distance_> > initialize() const {
        return initialize_known();
    }

    auto initialize_known() const {
        return std::make_unique<BruteforceSearcher<Index_, Data_, Distance_, DistanceMetric_> >(*this);
    }

public:
    void save(const std::filesystem::path& dir) const {
        quick_save(dir / "ALGORITHM", bruteforce_prebuilt_save_name, std::strlen(bruteforce_prebuilt_save_name));
        quick_save(dir / "DATA", my_data.data(), my_data.size());
        quick_save(dir / "NUM_OBS", &my_obs, 1);
        quick_save(dir / "NUM_DIM", &my_dim, 1);

        const auto distdir = dir / "DISTANCE";
        std::filesystem::create_directory(distdir);
        my_metric->save(distdir);
    }

    BruteforcePrebuilt(const std::filesystem::path& dir) {
        quick_load(dir / "NUM_OBS", &my_obs, 1);
        quick_load(dir / "NUM_DIM", &my_dim, 1);

        my_data.resize(sanisizer::product<I<decltype(my_data.size())> >(sanisizer::attest_gez(my_obs), my_dim));
        quick_load(dir / "DATA", my_data.data(), my_data.size());

        auto dptr = load_distance_metric_raw<Data_, Distance_>(dir / "DISTANCE");
        auto xptr = dynamic_cast<DistanceMetric_*>(dptr);
        if (xptr == NULL) {
            throw std::runtime_error("cannot cast the loaded distance metric to a DistanceMetric_");
        }
        my_metric.reset(xptr);
    }
};
/**
 * @endcond
 */

/**
 * @brief Perform a brute-force nearest neighbor search.
 *
 * The brute-force search computes all pairwise distances between data and query points to identify nearest neighbors of the latter.
 * It has quadratic complexity and is theoretically the worst-performing method;
 * however, it has effectively no overhead from constructing or querying indexing structures, 
 * potentially making it faster in cases where indexing provides little benefit (e.g., few data points, high dimensionality).
 *
 * @tparam Index_ Integer type for the indices.
 * @tparam Data_ Numeric type for the input and query data.
 * @tparam Distance_ Numeric type for the distances, usually floating-point.
 * @tparam Matrix_ Class of the input data matrix. 
 * This should satisfy the `Matrix` interface.
 * @tparam DistanceMetric_ Class implementing the distance metric calculation.
 * This should satisfy the `DistanceMetric` interface.
 */
template<
    typename Index_,
    typename Data_,
    typename Distance_,
    class Matrix_ = Matrix<Index_, Data_>,
    class DistanceMetric_ = DistanceMetric<Data_, Distance_>
>
class BruteforceBuilder final : public Builder<Index_, Data_, Distance_, Matrix_> {
public:
    /**
     * @param metric Pointer to a distance metric instance, e.g., `EuclideanDistance`.
     */
    BruteforceBuilder(std::shared_ptr<const DistanceMetric_> metric) : my_metric(std::move(metric)) {}

private:
    std::shared_ptr<const DistanceMetric_> my_metric;

public:
    Prebuilt<Index_, Data_, Distance_>* build_raw(const Matrix_& data) const {
        return build_known_raw(data);
    }

public:
    /**
     * Override to assist devirtualization.
     */
    auto build_known_raw(const Matrix_& data) const {
        std::size_t ndim = data.num_dimensions();
        const Index_ nobs = data.num_observations();
        auto work = data.new_known_extractor();

        // We assume that that vector::size_type <= size_t, otherwise data() wouldn't be a contiguous array.
        std::vector<Data_> store(sanisizer::product<typename std::vector<Data_>::size_type>(ndim, sanisizer::attest_gez(nobs)));
        for (Index_ o = 0; o < nobs; ++o) {
            std::copy_n(work->next(), ndim, store.data() + sanisizer::product_unsafe<std::size_t>(o, ndim));
        }

        return new BruteforcePrebuilt<Index_, Data_, Distance_, DistanceMetric_>(ndim, nobs, std::move(store), my_metric);
    }

    /**
     * Override to assist devirtualization.
     */
    auto build_known_unique(const Matrix_& data) const {
        return std::unique_ptr<I<decltype(*build_known_raw(data))> >(build_known_raw(data));
    }

    /**
     * Override to assist devirtualization.
     */
    auto build_known_shared(const Matrix_& data) const {
        return std::shared_ptr<I<decltype(*build_known_raw(data))> >(build_known_raw(data));
    }
};

}

#endif
