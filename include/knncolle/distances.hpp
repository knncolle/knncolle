#ifndef KNNCOLLE_DISTANCES_HPP
#define KNNCOLLE_DISTANCES_HPP

#include <cmath>

/**
 * @file distances.hpp
 *
 * @brief Classes for distance calculations.
 */

namespace knncolle {

/**
 * @brief Expectations for a distance calculation class.
 */
struct MockDistance {
    /**
     * The raw distance `r` for a distance `d` is defined so that `r(x, y) > r(x, z)` iff `d(x, y) > d(x, z)`.
     * `r(x, y)` is converted to `d(x, z)` via a monotonic transform in `normalize()`, and vice versa for `denormalize()`.
     * We separate out these two steps to avoid, e.g., a costly root operation for a Euclidean distance when only the relative values are of interest.
     *
     * @param x Pointer to the array containing the first vector.
     * @param y Pointer to the array containing the second vector.
     * @param num_dimensions Length of both vectors.
     *
     * @tparam Output_ Floating point type for the output distance.
     * @tparam DataX_ Floating point type for the first data vector.
     * @tparam DataY_ Floating point type for the second data vector.
     * @tparam Dim_ Integer type for the vector length.
     *
     * @return The raw distance between `x` and `y`.
     */
    template<typename Output_, typename DataX_, typename DataY_, typename Dim_>
    static Output_ raw_distance(const DataX_* x, const DataY_* y, Dim_ num_dimensions) {
        Output_ output = 0;
        for (Dim_ d = 0; d < num_dimensions; ++d, ++x, ++y) {
            auto delta = static_cast<Output_>(*x) - *y; // see below for comments.
            output += delta * delta;
        }
        return output;
    }

    /**
     * @tparam Output_ Floating point type for the output distance.
     * @param raw Raw distance.
     * @return The normalized distance.
     */
    template<typename Output_>
    static Output_ normalize(Output_ raw) {
        return raw;
    }

    /**
     * @tparam Output_ Floating point type for the output distance.
     * @param norm Normalized distance (i.e., the output of `normalize()`).
     * @return The denormalized distance (i.e., the input to `normalize()`).
     */
    template<typename Output_>
    static Output_ denormalize(Output_ norm) {
        return norm;
    }
};

/**
 * @brief Compute Euclidean distances between two input vectors.
 */
struct EuclideanDistance {
    /**
     *
     * @param x Pointer to the array containing the first vector.
     * @param y Pointer to the array containing the second vector.
     * @param num_dimensions Length of both vectors.
     *
     * @tparam Output_ Floating point type for the output distance.
     * @tparam DataX_ Floating point type for the first data vector.
     * @tparam DataY_ Floating point type for the second data vector.
     * @tparam Dim_ Integer type for the vector length.
     *
     * @return The squared Euclidean distance between `x` and `y`.
     */
    template<typename Output_, typename DataX_, typename DataY_, typename Dim_>
    static Output_ raw_distance(const DataX_* x, const DataY_* y, Dim_ num_dimensions) {
        Output_ output = 0;
        for (Dim_ d = 0; d < num_dimensions; ++d, ++x, ++y) {
            auto delta = static_cast<Output_>(*x) - static_cast<Output_>(*y); // casting to ensure consistent precision regardless of DataX_, DataY_.
            output += delta * delta;
        }
        return output;
    }

    /**
     * @tparam Output_ Floating point type for the output distance.
     * @param raw Squared Euclidean distance.
     * @return Euclidean distance.
     */
    template<typename Output_>
    static Output_ normalize(Output_ raw) {
        return std::sqrt(raw);
    }

    /**
     * @tparam Output_ Floating point type for the output distance.
     * @param norm Euclidean distance.
     * @return Squared Euclidean distance.
     */
    template<typename Output_>
    static Output_ denormalize(Output_ norm) {
        return norm * norm;
    }
};

/**
 * @brief Compute Manhattan distances between two input vectors.
 */
struct ManhattanDistance {
    /**
     *
     * @param x Pointer to the array containing the first vector.
     * @param y Pointer to the array containing the second vector.
     * @param num_dimensions Length of both vectors.
     *
     * @tparam Output_ Floating point type for the output distance.
     * @tparam DataX_ Floating point type for the first data vector.
     * @tparam DataY_ Floating point type for the second data vector.
     * @tparam Dim_ Integer type for the vector length.
     *
     * @return The Manhattan distance between `x` and `y`.
     */
    template<typename Output_, typename DataX_, typename DataY_, typename Dim_>
    static Output_ raw_distance(const DataX_* x, const DataY_* y, Dim_ num_dimensions) {
        Output_ output = 0;
        for (Dim_ d = 0; d < num_dimensions; ++d, ++x, ++y) {
            output += std::abs(static_cast<Output_>(*x) - static_cast<Output_>(*y)); // casting to ensure consistent precision regardless of DataX_, DataY_.
        }
        return output;
    }

    /**
     * @tparam Output_ Floating point type for the distance.
     * @param raw Manhattan distance.
     * @return `raw` with no modification.
     */
    template<typename Output_>
    static Output_ normalize(Output_ raw) {
        return raw;
    }

    /**
     * @tparam Output_ Floating point type for the distance.
     * @param norm Normalized distance.
     * @return `norm` with no modification.
     */
    template<typename Output_>
    static Output_ denormalize(Output_ norm) {
        return norm;
    }
};

}

#endif
