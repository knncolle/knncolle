#ifndef KNNCOLLE_DISTANCES_HPP
#define KNNCOLLE_DISTANCES_HPP

#include <cmath>
#include <cstddef>
#include <string>
#include <unordered_map>
#include <functional>
#include <cstring>
#include <memory>
#include <filesystem>

#include "utils.hpp"

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
 * @tparam Distance_ Numeric type for the output distance, usually floating-point.
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
     * An implementation of this method should create a `DISTANCE` file inside `dir` that contains the distance metric's name.
     * This should be an ASCII file with no newlines, where the metric name should follow the `<library>::<distance>` format, e.g., `knncolle::Euclidean`.
     * This will be used by `load_distance_metric_raw()` to determine the exact loader function to call. 
     *
     * Other than the `DISTANCE` file, each implementation may create any number of additional files of any format in `dir`.
     * We recommend that the name of each file/directory immediately starts with an upper case letter and is in all-capitals.
     * This allows applications to add more custom files without the risk of conflicts, e.g., by naming them without an upper-case letter. 
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
     * @param dir Path to a directory in which to save the index.
     * This should already exist.
     */
    virtual void save([[maybe_unused]] const std::filesystem::path& dir) const {
        throw std::runtime_error("saving is not supported");
    }
};

/**
 * Name for loading a `EuclideanDistance` in the `load_distance_registry()`.
 */
inline static constexpr const char* euclidean_distance_save_name = "knncolle::Euclidean";

/**
 * @brief Compute Euclidean distances between two input vectors.
 * @tparam Data_ Numeric type for the input data.
 * @tparam Distance_ Numeric type for the output distance, usually floating-point.
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

    void save(const std::filesystem::path& dir) const {
        quick_save(dir / "DISTANCE", euclidean_distance_save_name, std::strlen(euclidean_distance_save_name));
    }
    /**
     * @endcond
     */
};

/**
 * Name for loading a `ManhattanDistance` in the `load_distance_registry()`.
 */
inline static constexpr const char* manhattan_distance_save_name = "knncolle::Manhattan";

/**
 * @brief Compute Manhattan distances between two input vectors.
 *
 * @tparam Data_ Numeric type for the input data.
 * @tparam Distance_ Numeric type for the output distance, usually floating-point.
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

    void save(const std::filesystem::path& dir) const {
        quick_save(dir / "DISTANCE", manhattan_distance_save_name, std::strlen(manhattan_distance_save_name));
    }
    /**
     * @endcond
     */
};

/**
 * Distance loading function.
 * This accepts a path to a directory (see `DistanceMetric::save()`) and returns a pointer to a `Distance` instance.
 *
 * @tparam Data_ Numeric type for the input data.
 * @tparam Distance_ Numeric type for the output distance, usually floating-point.
 */
template<typename Data_, typename Distance_>
using LoadDistanceMetricFunction = std::function<DistanceMetric<Data_, Distance_>* (const std::filesystem::path&)>;


/**
 * @tparam Data_ Numeric type for the input data.
 * @tparam Distance_ Numeric type for the output distance, usually floating-point.
 *
 * @return Reference to a global map where the keys are distance metric names (see `DistanceMetric::save()`) and the values are distance loading functions.
 *
 * No loading functions are available when the global map is first initialized.
 * Users should call `register_load_euclidean_distance()` and/or `register_load_manhattan_distance()` to populate the map with loaders for distances they intend to support.
 */
template<typename Data_, typename Distance_>
inline std::unordered_map<std::string, LoadDistanceMetricFunction<Data_, Distance_> >& load_distance_metric_registry() {
    static std::unordered_map<std::string, LoadDistanceMetricFunction<Data_, Distance_> > registry; 
    return registry;
}

/**
 * Register a loading function for `EuclideanDistance` using `euclidean_distance_save_name`.
 *
 * @tparam Data_ Numeric type for the input data.
 * @tparam Distance_ Numeric type for the output distance, usually floating-point.
 */
template<typename Data_, typename Distance_>
void register_load_euclidean_distance() {
    auto& reg = load_distance_metric_registry<Data_, Distance_>();
    reg[euclidean_distance_save_name] = [](const std::filesystem::path&) -> DistanceMetric<Data_, Distance_>* { return new EuclideanDistance<Data_, Distance_>; };
}

/**
 * Register a loading function for `ManhattanDistance` using `manhattan_distance_save_name`.
 *
 * @tparam Data_ Numeric type for the input data.
 * @tparam Distance_ Numeric type for the output distance, usually floating-point.
 */
template<typename Data_, typename Distance_>
void register_load_manhattan_distance() {
    auto& reg = load_distance_metric_registry<Data_, Distance_>();
    reg[manhattan_distance_save_name] = [](const std::filesystem::path&) -> DistanceMetric<Data_, Distance_>* { return new ManhattanDistance<Data_, Distance_>; };
}

/**
 * @brief Exception for unknown distance metrics in `load_distance_metric_raw()`.
 *
 * This is thrown by `load_distance_metric_raw()` when it cannot find a function in the `load_distance_metric_registry()` for a particular distance metric.
 */
class LoadDistanceMetricNotFoundError final : public std::runtime_error {
public:
    /**
     * @cond
     */
    LoadDistanceMetricNotFoundError(std::string distance, std::filesystem::path path) : 
        std::runtime_error("cannot find a load_distance_metric_registry() function for '" + distance + "' at '" + path.string() + "'"),
        my_distance(std::move(distance)),
        my_path(std::move(path))
    {}
    /**
     * @endcond
     */

private:
    std::string my_distance;
    std::filesystem::path my_path;

public:
    /**
     * @return Name of the unknown neighbor search distance for the saved `Prebuilt` instance. 
     */
    const std::string& get_distance() const {
        return my_distance;
    }

    /**
     * @return Path to the `DISTANCE` file containing the distance name for the saved `Prebuilt` instance.
     */
    const std::filesystem::path& get_path() const {
        return my_path;
    }
};

/**
 * Load a distance metric from disk into a `Distance` object.
 *
 * @tparam Data_ Numeric type for the input data.
 * @tparam Distance_ Numeric type for the output distance, usually floating-point.
 *
 * @param dir Path to a directory containing a distance metric that was saved to disk by `DistanceMetric::save()`.
 *
 * @return Pointer to a `Distance` instance, created from the files at `dir`.
 * If no loading function is available for the saved distance, a `LoadDistanceMetricNotFoundError` is thrown.
 */
template<typename Data_, typename Distance_>
DistanceMetric<Data_, Distance_>* load_distance_metric_raw(const std::filesystem::path& dir) {
    const auto metric_path =  dir / "DISTANCE";
    const auto metric_name = quick_load_as_string(metric_path);

    const auto& reg = load_distance_metric_registry<Data_, Distance_>(); 
    auto it = reg.find(metric_name);
    if (it == reg.end()) {
        throw LoadDistanceMetricNotFoundError(metric_name, metric_path);
    }

    return (it->second)(dir);
}

}

#endif
