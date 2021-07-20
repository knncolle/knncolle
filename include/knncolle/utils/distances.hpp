#ifndef KNNCOLLE_DISTANCES_HPP
#define KNNCOLLE_DISTANCES_HPP
#include <cmath>

/**
 * @file distances.hpp
 *
 * @brief Classes for distance calculations.
 */

namespace knncolle {

namespace distances {

/**
 * @brief Compute Euclidean distances between two input vectors.
 */
struct Euclidean {
    /**
     * @param x Pointer to the array containing the first vector.
     * @param y Pointer to the array containing the second vector.
     * @param n Length of both vectors.
     *
     * @tparam ITYPE Integer type for the vector length.
     * @tparam DTYPE Floating point type for the output distance.
     * @tparam XTYPE Floating point type for the first data vector.
     * @tparam YTYPE Floating point type for the second data vector.
     *
     * @return The squared Euclidean distance between vectors.
     *
     * @note 
     * This should be passed through `normalize()` to obtain the actual Euclidean distance.
     * We separate out these two steps to avoid the costly root operation when only the relative values are of interest.
     */
    template<typename ITYPE = int, typename DTYPE = double, typename XTYPE = DTYPE, typename YTYPE = DTYPE>
    static DTYPE raw_distance(const XTYPE* x, const YTYPE* y, ITYPE n) {
        double output = 0;
        for (ITYPE i = 0; i < n; ++i, ++x, ++y) {
            output += ((*x) - (*y)) * ((*x) - (*y));
        }
        return output;
    }

    /**
     * @tparam DTYPE Floating point type for the distance.
     *
     * @param raw The value produced by `raw_distance()`.
     *
     * @return The square root of `raw`.
     */
    template<typename DTYPE = double>
    static DTYPE normalize(DTYPE raw) {
        return std::sqrt(raw);
    }
};

/**
 * @brief Compute Manhattan distances between two input vectors.
 */
struct Manhattan {
    /**
     * @tparam ITYPE Integer type for the vector length.
     * @tparam DTYPE Floating point type for the output distance.
     * @tparam XTYPE Floating point type for the first data vector.
     * @tparam YTYPE Floating point type for the second data vector.
     *
     * @param x Pointer to the array containing the first vector.
     * @param y Pointer to the array containing the second vector.
     * @param n Length of both vectors.
     *
     * @return The Manhattan distance between vectors.
     */
    template<typename ITYPE = int, typename DTYPE = double, typename XTYPE = DTYPE, typename YTYPE = DTYPE>
    static DTYPE raw_distance(const XTYPE* x, const YTYPE* y, ITYPE n) {
        DTYPE output = 0;
        for (ITYPE i = 0; i < n; ++i, ++x, ++y) {
            output += std::abs(*x - *y);
        }
        return output;
    }

    /**
     * @tparam DTYPE Floating point type for the distance.
     * @param raw The value produced by `raw_distance()`.
     *
     * @return `raw` with no modification.
     */
    template<typename DTYPE = double>
    static DTYPE normalize(DTYPE raw) {
        return raw;
    }
};

}

}

#endif
