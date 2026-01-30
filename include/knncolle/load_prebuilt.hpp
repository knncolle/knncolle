#ifndef KNNCOLLE_LOAD_PREBUILT_HPP
#define KNNCOLLE_LOAD_PREBUILT_HPP

#include <string>
#include <unordered_map>
#include <functional>
#include <fstream>
#include <memory>

#include "Prebuilt.hpp"
#include "Bruteforce.hpp"
#include "Vptree.hpp"
#include "L2Normalized.hpp"

/**
 * @file load_prebuilt.hpp
 * @brief Load prebuilt search indices from disk.
 */

namespace knncolle {

/**
 * Prebuilt loading function.
 * This accepts a directory (see `Prebuilt::save()`) and returns a pointer to a `Prebuilt` instance.
 *
 * @tparam Index_ Integer type for the observation indices.
 * @tparam Data_ Numeric type for the query data.
 * @tparam Distance_ Numeric type for the distances, usually floating-point.
 */
template<typename Index_, typename Data_, typename Distance_>
using LoadPrebuiltFunction = std::function<Prebuilt<Index_, Data_, Distance_>* (const std::filesystem::path&)>;

/**
 * @tparam Index_ Integer type for the observation indices.
 * @tparam Data_ Numeric type for the query data.
 * @tparam Distance_ Numeric type for the distances, usually floating-point.
 *
 * @return Reference to a global map where the keys are algorithm names (see `Prebuilt::save()`) and the values are the prebuilt loading functions.
 *
 * No loading functions are available when the global is first initialized.
 * Users should call `register_load_bruteforce_prebuilt()`, `register_load_vptree_prebuilt()`, etc. to populate the map with loaders for algorithms they intend to support.
 * For L2-normalized indices, a loader function can be defined with `load_l2normalized_prebuilt()`.
 */
template<typename Index_, typename Data_, typename Distance_>
inline std::unordered_map<std::string, LoadPrebuiltFunction<Index_, Data_, Distance_> >& load_prebuilt_registry() {
    static std::unordered_map<std::string, LoadPrebuiltFunction<Index_, Data_, Distance_> > registry;
    return registry;
}

/**
 * @brief Exception for unknown search algorithms in `load_prebuilt_raw()`.
 *
 * This is thrown by `load_prebuilt_raw()` and related functions when they cannot find a function in the `load_prebuilt_registry()` for a particular algorithm.
 */
class LoadPrebuiltNotFoundError final : public std::runtime_error {
public:
    /**
     * @cond
     */
    LoadPrebuiltNotFoundError(std::string algorithm, std::filesystem::path path) : 
        std::runtime_error("cannot find a load_prebuilt_registry() function for '" + algorithm + "' at '" + path.string() + "'"),
        my_algorithm(std::move(algorithm)),
        my_path(std::move(path))
    {}
    /**
     * @endcond
     */

private:
    std::string my_algorithm;
    std::filesystem::path my_path;

public:
    /**
     * @return Name of the unknown neighbor search algorithm for the saved `Prebuilt` instance. 
     */
    const std::string& get_algorithm() const {
        return my_algorithm;
    }

    /**
     * @return Path to the `ALGORITHM` file containing the algorithm name for the saved `Prebuilt` instance.
     */
    const std::filesystem::path& get_path() const {
        return my_path;
    }
};

/**
 * Load a neighbor search index from disk into a `Prebuilt` object.
 * This should be called with the same template parameters as the `Prebuilt` interface from which `Prebuilt::save()` was called.
 * It is expected that `load_prebuilt_raw()` should create an object that is "equivalent" to the object that was saved with `save()`,
 * i.e., any neighbor search results should be the same across the original and reloaded `Prebuilt` object.
 *
 * @tparam Index_ Integer type for the observation indices.
 * @tparam Data_ Numeric type for the query data.
 * @tparam Distance_ Numeric type for the distances, usually floating-point.
 *
 * @param dir Path to a directory containing a prebuilt index that was saved to disk by `Prebuilt::save()`.
 *
 * @return Pointer to a `Prebuilt` instance, created from the files at `dir`.
 * If no loading function can be found for the search algorithm, a `LoadPrebuiltNotFoundError` is thrown.
 */
template<typename Index_, typename Data_, typename Distance_>
Prebuilt<Index_, Data_, Distance_>* load_prebuilt_raw(const std::filesystem::path& dir) {
    const auto alg_path = dir / "ALGORITHM";
    const auto algorithm = quick_load_as_string(alg_path);

    const auto& reg = load_prebuilt_registry<Index_, Data_, Distance_>(); 
    auto it = reg.find(algorithm);
    if (it == reg.end()) {
        throw LoadPrebuiltNotFoundError(algorithm, alg_path);
    }

    return (it->second)(dir);
}

/**
 * Load a neighbor search index from disk into a `Prebuilt` object. 
 * This should be called with the same template parameters as the `Prebuilt` interface from which `save()` was called.
 *
 * @tparam Index_ Integer type for the observation indices.
 * @tparam Data_ Numeric type for the query data.
 * @tparam Distance_ Numeric type for the distances, usually floating-point.
 *
 * @param dir Path to a directory containing a prebuilt index that was saved to disk by `Prebuilt::save()`.
 *
 * @return Unique pointer to a `Prebuilt` instance, created from the files at `dir`.
 * This uses the return value of `load_prebuilt_raw()`. 
 */
template<typename Index_, typename Data_, typename Distance_>
std::unique_ptr<Prebuilt<Index_, Data_, Distance_> > load_prebuilt_unique(const std::filesystem::path& dir) {
    return std::unique_ptr<Prebuilt<Index_, Data_, Distance_> >(load_prebuilt_raw<Index_, Data_, Distance_>(dir));
}

/**
 * Load a neighbor search index from disk into a `Prebuilt` object. 
 * This should be called with the same template parameters as the `Prebuilt` interface from which `save()` was called.
 *
 * @tparam Index_ Integer type for the observation indices.
 * @tparam Data_ Numeric type for the query data.
 * @tparam Distance_ Numeric type for the distances, usually floating-point.
 *
 * @param dir Path to a directory containing a prebuilt index that was saved to disk by `Prebuilt::save()`.
 *
 * @return Shared pointer to a `Prebuilt` instance, created from the files at `dir`.
 * This uses the return value of `load_prebuilt_raw()`. 
 */
template<typename Index_, typename Data_, typename Distance_>
std::shared_ptr<Prebuilt<Index_, Data_, Distance_> > load_prebuilt_shared(const std::filesystem::path& dir) {
    return std::shared_ptr<Prebuilt<Index_, Data_, Distance_> >(load_prebuilt_raw<Index_, Data_, Distance_>(dir));
}

/**
 * @brief Template type of a saved L2-normalized index.
 *
 * Instances are typically created by `load_l2normalized_prebuilt_types()`.
 */
struct L2NormalizedPrebuiltTypes {
    /**
     * Type of the L2-normalized data.
     */
    NumericType normalized;
};

/**
 * @param dir Path to a directory in which a prebuilt L2-normalized index was saved.
 * Files should have been generated by the `Prebuilt::save()` method of the L2-normalized `Prebuilt` subclass instance.
 *
 * @return Template types of the saved instance of a `Prebuilt` L2-normalized subclass.
 * This is typically used to choose template parameters for `load_l2normalized_prebuilt()`.
 */
inline L2NormalizedPrebuiltTypes load_l2normalized_prebuilt_types(const std::filesystem::path& dir) {
    L2NormalizedPrebuiltTypes config;
    quick_load(dir / "NORMALIZED", &(config.normalized), 1);
    return config;
}

/**
 * Load an L2-normalized search index (i.e., a `Prebuilt` created by `L2NormalizedBuilder`) from its on-disk representation.
 * This is not provided in the registry by default as its depends on the application's set of the acceptable `Normalized_` types.
 * The `Normalized_` type in the saved index can be retrived by `load_l2normalized_prebuilt_types()`.
 *
 * @tparam Index_ Integer type for the indices.
 * @tparam Data_ Numeric type for the input and query data.
 * @tparam Distance_ Numeric type for the distances, usually floating-point.
 * @tparam Normalized_ Floating-point type for the L2-normalized data.
 *
 * @param dir Path to a directory in which a prebuilt L2-normalized index was saved.
 * Files should have been generated by the `Prebuilt::save()` method of the L2-normalized `Prebuilt` subclass instance.
 * 
 * @return Pointer to an L2-normalized `Prebuilt` instance.
 */
template<typename Index_, typename Data_, typename Distance_, typename Normalized_>
Prebuilt<Index_, Data_, Distance_>* load_l2normalized_prebuilt(const std::filesystem::path& dir) {
    return new L2NormalizedPrebuilt<Index_, Data_, Distance_, Normalized_>(dir);
}

/**
 * Load an brute-force search index (i.e., a `Prebuilt` created by `BruteforceBuilder`) from its on-disk representation.
 * This is not provided in the registry by default as its depends on the application's choice of `DistanceMetric_`.
 *
 * @tparam Index_ Integer type for the observation indices.
 * @tparam Data_ Numeric type for the query data.
 * @tparam Distance_ Floating-point type for the distances.
 * @tparam DistanceMetric_ Class implementing the distance metric calculation.
 * This should satisfy the `DistanceMetric` interface.
 *
 * @param dir Path to a directory in which a prebuilt brute-force index was saved.
 * Files should have been generated by the `Prebuilt::save()` method of the brute-force `Prebuilt` subclass instance.
 *
 * @return Pointer to a brute-force `Prebuilt` instance.
 */
template<typename Index_, typename Data_, typename Distance_, class DistanceMetric_ = DistanceMetric<Data_, Distance_> >
Prebuilt<Index_, Data_, Distance_>* load_bruteforce_prebuilt(const std::filesystem::path& dir) {
    return new BruteforcePrebuilt<Index_, Data_, Distance_, DistanceMetric_>(dir);
}

/**
 * Load a VP-tree search index (i.e., a `Prebuilt` created by `VptreeBuilder`) from its on-disk representation.
 * This is not provided in the registry by default as its depends on the application's choice of `DistanceMetric_`.
 *
 * @tparam Index_ Integer type for the observation indices.
 * @tparam Data_ Numeric type for the query data.
 * @tparam Distance_ Floating-point type for the distances.
 * @tparam DistanceMetric_ Class implementing the distance metric calculation.
 * This should satisfy the `DistanceMetric` interface.
 *
 * @param dir Path to a directory in which a prebuilt VP-tree index was saved.
 * Files should have been generated by the `Prebuilt::save()` method of the VP-tree `Prebuilt` subclass instance.
 *
 * @return Pointer to a VP-tree `Prebuilt` instance.
 */
template<typename Index_, typename Data_, typename Distance_, class DistanceMetric_ = DistanceMetric<Data_, Distance_> >
Prebuilt<Index_, Data_, Distance_>* load_vptree_prebuilt(const std::filesystem::path& dir) {
    return new VptreePrebuilt<Index_, Data_, Distance_, DistanceMetric_>(dir);
}

}

#endif
