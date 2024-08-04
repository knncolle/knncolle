#ifndef KNNCOLLE_PREBUILT_HPP
#define KNNCOLLE_PREBUILT_HPP

#include <memory>
#include "Searcher.hpp"

/**
 * @file Prebuilt.hpp
 *
 * @brief Interface for prebuilt nearest-neighbor indices.
 */

namespace knncolle {

/**
 * @brief Interface for prebuilt nearest-neighbor search indices.
 *
 * Instances of `Prebuilt` subclasses are typically constructed with `Builder::build_raw()`.
 * Note that a `Prebuilt` instance may outlive the `Builder` object used to generate it, so the former should not hold any references to the latter.
 *
 * @tparam Dim_ Integer type for the number of dimensions.
 * For the output of `Builder::build_raw()`, this is set to `Matrix_::dimension_type`.
 * @tparam Index_ Integer type for the indices.
 * For the output of `Builder::build_raw()`, this is set to `Matrix_::index_type`.
 * @tparam Float_ Floating point type for the query data and output distances.
 */
template<typename Dim_, typename Index_, typename Float_>
class Prebuilt {
public:
    /**
     * @return Number of observations in the dataset to be searched.
     */
    virtual Index_ num_observations() const = 0;
    
    /**
     * @return Number of dimensions.
     */
    virtual Dim_ num_dimensions() const = 0;

    /**
     * @cond
     */
    virtual ~Prebuilt() = default;
    /**
     * @endcond
     */

public:
    /**
     * Create a `Searcher` for searching the index.
     * @return Pointer to a `Searcher` instance.
     */
    virtual std::unique_ptr<Searcher<Index_, Float_> > initialize() const = 0;
};

}

#endif
