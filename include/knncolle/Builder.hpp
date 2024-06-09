#ifndef KNNCOLLE_BUILDER_HPP
#define KNNCOLLE_BUILDER_HPP

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
template<class MockMatrix_, typename Float_>
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
    Prebuilt<typename Matrix_::dimension_type, typename Matrix_::index_type, Float_>* build(const MockMatrix_& input) const = default;
};

}

#endif
