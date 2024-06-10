#ifndef KNNCOLLE_BUILDER_HPP
#define KNNCOLLE_BUILDER_HPP

#include "Prebuilt.hpp"
#include <memory>

/**
 * @file Builder.hpp
 *
 * @brief Interface to build nearest-neighbor indices.
 */

namespace knncolle {

/**
 * @brief Interface to build nearest-neighbor search indices.
 *
 * @tparam Matrix_ Matrix-like type that satisfies the `MockMatrix` interface.
 * @tparam Float_ Floating point type for the query data and output distances.
 */
template<class Matrix_, typename Float_>
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
     * @param data Matrix-like object (see `MockMatrix`) containing observations in columns and dimensions in rows.
     * @return Pointer to a pre-built search index.
     */
    virtual Prebuilt<typename Matrix_::dimension_type, typename Matrix_::index_type, Float_>* build_raw(const Matrix_& data) const = 0;

    /**
     * @param data Matrix-like object (see `MockMatrix`) containing observations in columns and dimensions in rows.
     * @return Shared pointer to a pre-built search index.
     */
    std::shared_ptr<Prebuilt<typename Matrix_::dimension_type, typename Matrix_::index_type, Float_> > build_shared(const Matrix_& data) const {
        return std::shared_ptr<Prebuilt<typename Matrix_::dimension_type, typename Matrix_::index_type, Float_> >(build_raw(data));
    }

    /**
     * @param data Matrix-like object (see `MockMatrix`) containing observations in columns and dimensions in rows.
     * @return Unique pointer to a pre-built search index.
     */
    std::unique_ptr<Prebuilt<typename Matrix_::dimension_type, typename Matrix_::index_type, Float_> > build_unique(const Matrix_& data) const {
        return std::unique_ptr<Prebuilt<typename Matrix_::dimension_type, typename Matrix_::index_type, Float_> >(build_raw(data));
    }
};

}

#endif
