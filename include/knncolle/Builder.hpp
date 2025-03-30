#ifndef KNNCOLLE_BUILDER_HPP
#define KNNCOLLE_BUILDER_HPP

#include "Prebuilt.hpp"
#include "Matrix.hpp"
#include <memory>
#include <utility>

/**
 * @file Builder.hpp
 *
 * @brief Interface to build nearest-neighbor indices.
 */

namespace knncolle {

/**
 * @brief Interface to build nearest-neighbor search indices.
 *
 * @tparam Dim_ Integer type for the number of dimensions.
 * @tparam Index_ Integer type for the indices.
 * @tparam Float_ Floating point type for the query data and output distances.
 * @tparam Matrix_ Class that satisfies the `Matrix` interface.
 */
template<typename Dim_, typename Index_, typename Data_, typename Float_, class Matrix_ = Matrix<Dim_, Index_, Float_> >
class Builder {
public:
    /**
     * @cond
     */
    virtual ~Builder() = default;
    /**
     * @endcond
     */

    /**
     * @param data Object satisfying the `Matrix` interface, containing observations in columns and dimensions in rows.
     * @return Pointer to a pre-built search index.
     */
    virtual Prebuilt<Dim_, Index_, Float_>* build_raw(const Matrix_& data) const = 0;

    /**
     * @param data Object satisfying the `Matrix` interface, containing observations in columns and dimensions in rows.
     * @return Shared pointer to a pre-built search index.
     */
    std::shared_ptr<Prebuilt<Dim_, Index_, Float_> > build_shared(const Matrix_& data) const {
        return std::shared_ptr<Prebuilt<Dim_, Index_, Float_> >(build_raw(data));
    }

    /**
     * @param data Object satisfying the `Matrix` interface, containing observations in columns and dimensions in rows.
     * @return Unique pointer to a pre-built search index.
     */
    std::unique_ptr<Prebuilt<Dim_, Index_, Float_> > build_unique(const Matrix_& data) const {
        return std::unique_ptr<Prebuilt<Dim_, Index_, Float_> >(build_raw(data));
    }
};

}

#endif
