#ifndef KNNCOLLE_BUILDER_HPP
#define KNNCOLLE_BUILDER_HPP

#include "Prebuilt.hpp"
#include "Matrix.hpp"
#include <memory>
#include <utility>
#include <type_traits>

/**
 * @file Builder.hpp
 *
 * @brief Interface to build nearest-neighbor indices.
 */

namespace knncolle {

/**
 * @brief Interface to build nearest-neighbor search indices.
 *
 * @tparam Index_ Integer type for the observation indices.
 * @tparam Data_ Numeric type for the input and query data.
 * @tparam Distance_ Floating point type for the distances.
 * @tparam Matrix_ Class of the input data matrix. 
 * This should satisfy the `Matrix` interface.
 */
template<typename Index_, typename Data_, typename Distance_, class Matrix_ = Matrix<Index_, Data_> >
class Builder {
public:
    /**
     * @cond
     */
    // Rule of 5 all of this.
    Builder() = default;
    Builder(Builder&&) = default;
    Builder(const Builder&) = default;
    Builder& operator=(Builder&&) = default;
    Builder& operator=(const Builder&) = default;
    virtual ~Builder() = default;

    static_assert(std::is_same<decltype(std::declval<Matrix_>().num_observations()), Index_>::value);
    static_assert(std::is_same<typename std::remove_pointer<decltype(std::declval<Matrix_>().new_extractor()->next())>::type, const Data_>::value);
    /**
     * @endcond
     */

public:
    /**
     * @param data Object satisfying the `Matrix` interface, containing observations in columns and dimensions in rows.
     * @return Pointer to a pre-built search index.
     */
    virtual Prebuilt<Index_, Data_, Distance_>* build_raw(const Matrix_& data) const = 0;

    /**
     * @param data Object satisfying the `Matrix` interface, containing observations in columns and dimensions in rows.
     * @return Shared pointer to a pre-built search index.
     */
    std::shared_ptr<Prebuilt<Index_, Data_, Distance_> > build_shared(const Matrix_& data) const {
        return std::shared_ptr<Prebuilt<Index_, Data_, Distance_> >(build_raw(data));
    }

    /**
     * @param data Object satisfying the `Matrix` interface, containing observations in columns and dimensions in rows.
     * @return Unique pointer to a pre-built search index.
     */
    std::unique_ptr<Prebuilt<Index_, Data_, Distance_> > build_unique(const Matrix_& data) const {
        return std::unique_ptr<Prebuilt<Index_, Data_, Distance_> >(build_raw(data));
    }
};

}

#endif
