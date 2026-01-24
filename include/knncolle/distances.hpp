#ifndef KNNCOLLE_DISTANCES_HPP
#define KNNCOLLE_DISTANCES_HPP

#include <cmath>
#include <cstddef>
#include <string>
#include <unordered_map>
#include <functional>
#include <fstream>
#include <memory>

/**
 * @file distances.hpp
 *
 * @brief Classes for distance calculations.
 */

namespace knncolle {

/**
 * @brief Interface for a distance metric.
 *
 * @tparam Data_ Numeric type for the input data.
 * @tparam Distance_ Floating-point type for the output distance.
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

public:
    /**
     * Save the distance metric to disk, to be reloaded with `load_distance_metric_raw()` and friends.
     *
     * An implementation of this method should create a `<prefix>DISTANCE` file that contains the distance metric's name.
     * This should be an ASCII file with no newlines, where the method name should follow the `<library>::<distance>` format, e.g., `knncolle::Euclidean`.
     * This will be used by `load_distance_metric_raw()` to determine the exact loader function to call. 
     * Other than the `DISTANCE` file, each implementation may create any number of additional files of any format, as long as they start with `prefix`.
     *
     * An implementation of this method is not required to use portable file formats.
     * `load_distance_metric_raw()` is only expected to work on the same system (i.e., architecture, compiler, compilation settings) that was used for the `save()` call.
     * Any additional portability is at the discretion of the implementation, e.g., it is common to assume IEEE floating-point and two's-complement integers.
     *
     * An implementation of this method is not required to create files that are readable by different versions of the implementation. 
     * Thus, the files created by this method are generally unsuitable for archival storage.
     * However, implementations are recommended to at least provide enough information to throw an exception if an incompatible version of `load_distance_metric_raw()` is used.
     *
     * If a subclass does not implement this method, an error is thrown by default.
     *
     * @param prefix Prefix of the file path(s) in which to save the index.
     * All files created by this method should start with this prefix. 
     * Any directories required to write a file starting with `prefix` should already have been created.
     */
    virtual void save([[maybe_unused]] const std::string& prefix) const {
        throw std::runtime_error("saving is not supported");
    }
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

    void save(const std::string& prefix) const {
        const std::string method_name = "knncolle::Euclidean";
        std::ofstream output(prefix + "DISTANCE");
        output.write(method_name.c_str(), method_name.size());
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

    void save(const std::string& prefix) const {
        const std::string method_name = "knncolle::Manhattan";
        std::ofstream output(prefix + "DISTANCE");
        output.write(method_name.c_str(), method_name.size());
    }
    /**
     * @endcond
     */
};

/**
 * Distance loading function.
 * This accepts a file path prefix (see `DistanceMetric::save()`) and returns a pointer to a `Distance` instance.
 *
 * @tparam Data_ Numeric type for the input data.
 * @tparam Distance_ Floating-point type for the output distance.
 */
template<typename Data_, typename Distance_>
using LoadDistanceMetricFunction = std::function<DistanceMetric<Data_, Distance_>* (const std::string&)>;

/**
 * @cond
 */
template<typename Data_, typename Distance_>
auto default_distance_metric_registry() {
    std::unordered_map<std::string, LoadDistanceMetricFunction<Data_, Distance_> > registry;
    registry["knncolle::Euclidean"] = [](const std::string&) -> DistanceMetric<Data_, Distance_>* { return new EuclideanDistance<Data_, Distance_>; };
    registry["knncolle::Manhattan"] = [](const std::string&) -> DistanceMetric<Data_, Distance_>* { return new ManhattanDistance<Data_, Distance_>; };
    return registry;
}
/**
 * @endcond
 */

/**
 * @tparam Data_ Numeric type for the input data.
 * @tparam Distance_ Floating-point type for the output distance.
 *
 * @return Reference to a global map of method names (see `DistanceMetric::save()`) to loading functions.
 */
template<typename Data_, typename Distance_>
inline std::unordered_map<std::string, LoadDistanceMetricFunction<Data_, Distance_> >& load_distance_metric_registry() {
    static std::unordered_map<std::string, LoadDistanceMetricFunction<Data_, Distance_> > registry = default_distance_metric_registry<Data_, Distance_>();
    return registry;
}

/**
 * Load a distance metric from disk into a `Distance` object.
 *
 * @tparam Data_ Numeric type for the input data.
 * @tparam Distance_ Floating-point type for the output distance.
 *
 * @param prefix File path prefix for a distance index that was saved to disk by `DistanceMetric::save()`.
 *
 * @return Pointer to a `Distance` instance, created from the files at `prefix`.
 */
template<typename Data_, typename Distance_>
DistanceMetric<Data_, Distance_>* load_distance_metric_raw(const std::string& prefix) {
    const auto meth_path = prefix + "DISTANCE";
    std::ifstream input(meth_path);
    std::string method( (std::istreambuf_iterator<char>(input)), (std::istreambuf_iterator<char>()) );

    const auto& reg = load_distance_metric_registry<Data_, Distance_>(); 
    auto it = reg.find(method);
    if (it == reg.end()) {
        throw std::runtime_error("cannot find load_distance_metric method for '" + method + "' at '" + meth_path + "'");
    }

    return (it->second)(prefix);
}

}

#endif
