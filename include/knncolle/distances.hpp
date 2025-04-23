#ifndef KNNCOLLE_DISTANCES_HPP
#define KNNCOLLE_DISTANCES_HPP

#include <cmath>
#include <cstddef>

/**
 * @file distances.hpp
 *
 * @brief Classes for distance calculations.
 */

namespace knncolle {

/**
 * @brief Interface for a distance metric.
 *
 * @tparam Distance_ Floating-point type for the output distance.
 * @tparam Data_ Numeric type for the input data.
 */
template<typename Data_, typename Distance_>
class DistanceMetric {
public:
    /**
     * @cond
     */
    DistanceMetric() = default;
    DistanceMetric(const DistanceMetric&) = default;
    DistanceMetric(DistanceMetric&&) = default;
    DistanceMetric& operator=(const DistanceMetric&) = default;
    DistanceMetric& operator=(DistanceMetric&&) = default;
    virtual ~DistanceMetric() = default;
    /**
     * @endcond
     */

public:
    /**
     * The raw distance `r` for a distance `d` is defined so that `r(x, y) > r(x, z)` iff `d(x, y) > d(x, z)`.
     * `r(x, y)` is converted to `d(x, z)` via a monotonic transform in `normalize()`, and vice versa for `denormalize()`.
     * We separate out these two steps to avoid, e.g., a costly root operation for a Euclidean distance when only the relative values are of interest.
     *
     * @param x Pointer to the array containing the first vector.
     * @param y Pointer to the array containing the second vector.
     * @param num_dimensions Length of both vectors.
     *
     * @return The raw distance between `x` and `y`.
     */
    virtual Distance_ raw(std::size_t num_dimensions, const Data_* x, const Data_* y) const = 0;

    /**
     * @param raw Raw distance.
     * @return The normalized distance.
     */
    virtual Distance_ normalize(Distance_ raw) const = 0;

    /**
     * @param norm Normalized distance (i.e., the output of `normalize()`).
     * @return The denormalized distance (i.e., the input to `normalize()`).
     */
    virtual Distance_ denormalize(Distance_ norm) const = 0;
};

/**
 * @brief Compute Euclidean distances between two input vectors.
 *
 * @tparam Distance_ Floating-point type for the output distance.
 * @tparam Data_ Numeric type for the input data.
 */
template<typename Data_, typename Distance_>
class EuclideanDistance final : public DistanceMetric<Data_, Distance_> {
public:
    /**
     * @cond
     */
    Distance_ raw(std::size_t num_dimensions, const Data_* x, const Data_* y) const {
        Distance_ output = 0;
        for (std::size_t d = 0; d < num_dimensions; ++d) {
            auto delta = static_cast<Distance_>(x[d]) - static_cast<Distance_>(y[d]); // casting to ensure consistent precision/signedness regardless of Data_.
            output += delta * delta;
        }
        return output;
    }

    Distance_ normalize(Distance_ raw) const {
        return std::sqrt(raw);
    }

    Distance_ denormalize(Distance_ norm) const {
        return norm * norm;
    }
    /**
     * @endcond
     */
};


/**
 * @brief Compute Manhattan distances between two input vectors.
 *
 * @tparam Distance_ Floating-point type for the output distance.
 * @tparam Data_ Numeric type for the input data.
 */
template<typename Data_, typename Distance_>
class ManhattanDistance final : public DistanceMetric<Data_, Distance_> {
public:
    /**
     * @cond
     */
    Distance_ raw(std::size_t num_dimensions, const Data_* x, const Data_* y) const {
        Distance_ output = 0;
        for (std::size_t d = 0; d < num_dimensions; ++d) {
            auto delta = static_cast<Distance_>(x[d]) - static_cast<Distance_>(y[d]); // casting to ensure consistent precision/signedness regardless of Data_.
            output += std::abs(delta);
        }
        return output;
    }

    Distance_ normalize(Distance_ raw) const {
        return raw;
    }

    Distance_ denormalize(Distance_ norm) const {
        return norm;
    }
    /**
     * @endcond
     */
};

}

#endif
