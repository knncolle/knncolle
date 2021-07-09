#ifndef KNNCOLLE_DISTANCES_HPP
#define KNNCOLLE_DISTANCES_HPP
#include <cmath>

/**
 * @file distances.hpp
 *
 * Classes for distance calculations.
 */

namespace knncolle {

namespace distances {

/**
 * @brief Compute Euclidean distances between two input vectors.
 *
 * @tparam ITYPE Integer type for the vector length.
 * @tparam DTYPE Floating point type for the data.
 */
template<typename ITYPE = int, typename DTYPE = double>
struct Euclidean {
    /**
     * @param x Pointer to the array containing the first vector.
     * @param y Pointer to the array containing the second vector.
     * @param n Length of both vectors.
     *
     * @return The squared Euclidean distance between vectors.
     *
     * @note 
     * This should be passed through `normalize()` to obtain the actual Euclidean distance.
     * We separate out these two steps to avoid the costly root operation when only the relative values are of interest.
     */
    static DTYPE raw_distance(const DTYPE* x, const DTYPE* y, ITYPE n) {
        double output = 0;
        for (ITYPE i = 0; i < n; ++i, ++x, ++y) {
            output += ((*x) - (*y)) * ((*x) - (*y));
        }
        return output;
    }

    /**
     * @param raw The value produced by `raw_distance()`.
     *
     * @return The square root of `raw`.
     */
    static DTYPE normalize(DTYPE raw) {
        return std::sqrt(raw);
    }
};

/**
 * @brief Compute Manhattan distances between two input vectors.
 *
 * @tparam ITYPE Integer type for the vector length.
 * @tparam DTYPE Floating point type for the distances.
 */
template<typename ITYPE = int, typename DTYPE = double>
struct Manhattan {
    /**
     * @param x Pointer to the array containing the first vector.
     * @param y Pointer to the array containing the second vector.
     * @param n Length of both vectors.
     *
     * @return The Manhattan distance between vectors.
     */
    static double raw_distance(const DTYPE* x, const DTYPE* y, ITYPE n) {
        double output = 0;
        for (ITYPE i = 0; i < n; ++i, ++x, ++y) {
            output += std::abs(*x - *y);
        }
        return output;
    }

    /**
     * @param raw The value produced by `raw_distance()`.
     *
     * @return `raw` with no modification.
     */
    static double normalize(double raw) {
        return raw;
    }
};

}

}

#endif
