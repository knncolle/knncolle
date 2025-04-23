#ifndef KNNCOLLE_PREBUILT_HPP
#define KNNCOLLE_PREBUILT_HPP

#include <memory>
#include <cstddef>

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
 * @tparam Index_ Integer type for the observation indices.
 * @tparam Data_ Numeric type for the query data.
 * @tparam Distance_ Floating point type for the distances.
 */
template<typename Index_, typename Data_, typename Distance_>
class Prebuilt {
public:
    /**
     * @cond
     */
    Prebuilt() = default;
    Prebuilt(const Prebuilt&) = default;
    Prebuilt(Prebuilt&&) = default;
    Prebuilt& operator=(const Prebuilt&) = default;
    Prebuilt& operator=(Prebuilt&&) = default;
    virtual ~Prebuilt() = default;
    /**
     * @endcond
     */

public:
    /**
     * @return Number of observations in the dataset to be searched.
     */
    virtual Index_ num_observations() const = 0;
    
    /**
     * @return Number of dimensions.
     */
    virtual std::size_t num_dimensions() const = 0;

public:
    /**
     * Create a `Searcher` for searching the index.
     * @return Pointer to a `Searcher` instance.
     */
    virtual std::unique_ptr<Searcher<Index_, Data_, Distance_> > initialize() const = 0;
};

}

#endif
