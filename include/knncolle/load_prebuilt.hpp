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
 * This accepts a file path prefix (see `Prebuilt::save()`) and returns a pointer to a `Prebuilt` instance.
 *
 * @tparam Index_ Integer type for the observation indices.
 * @tparam Data_ Numeric type for the query data.
 * @tparam Distance_ Floating point type for the distances.
 */
template<typename Index_, typename Data_, typename Distance_>
using LoadPrebuiltFunction = std::function<Prebuilt<Index_, Data_, Distance_>* (const std::string&)>;

/**
 * @tparam Index_ Integer type for the observation indices.
 * @tparam Data_ Numeric type for the query data.
 * @tparam Distance_ Floating point type for the distances.
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
 * Register a loading function for the brute-force `Prebuilt` indices using `bruteforce_prebuilt_save_name`.
 *
 * @tparam Index_ Integer type for the observation indices.
 * @tparam Data_ Numeric type for the query data.
 * @tparam Distance_ Floating point type for the distances.
 */
template<typename Index_, typename Data_, typename Distance_>
void register_load_bruteforce_prebuilt() {
    auto& reg = load_prebuilt_registry<Index_, Data_, Distance_>();
    reg[bruteforce_prebuilt_save_name] = [](const std::string& prefix) -> Prebuilt<Index_, Data_, Distance_>* {
        return new BruteforcePrebuilt<Index_, Data_, Distance_, DistanceMetric<Data_, Distance_> >(prefix);
    };
}

/**
 * Register a loading function for the VP-tree `Prebuilt` indices using `vptree_prebuilt_save_name`.
 *
 * @tparam Index_ Integer type for the observation indices.
 * @tparam Data_ Numeric type for the query data.
 * @tparam Distance_ Floating point type for the distances.
 */
template<typename Index_, typename Data_, typename Distance_>
void register_load_vptree_prebuilt() {
    auto& reg = load_prebuilt_registry<Index_, Data_, Distance_>();
    reg[vptree_prebuilt_save_name] = [](const std::string& prefix) -> Prebuilt<Index_, Data_, Distance_>* {
        return new VptreePrebuilt<Index_, Data_, Distance_, DistanceMetric<Data_, Distance_> >(prefix);
    };
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
    LoadPrebuiltNotFoundError(std::string algorithm, std::string path) : 
        std::runtime_error("cannot find a load_prebuilt_registry() function for '" + algorithm + "' at '" + path + "'"),
        my_algorithm(std::move(algorithm)),
        my_path(std::move(path))
    {}
    /**
     * @endcond
     */

private:
    std::string my_algorithm, my_path;

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
    const std::string& get_path() const {
        return my_path;
    }
};

/**
 * Load a neighbor search index from disk into a `Prebuilt` object.
 * This should be called with the same template parameters as the `Prebuilt` interface from which `save()` was called.
 *
 * @tparam Index_ Integer type for the observation indices.
 * @tparam Data_ Numeric type for the query data.
 * @tparam Distance_ Floating point type for the distances.
 *
 * @param prefix File path prefix for a prebuilt index that was saved to disk by `Prebuilt::save()`.
 *
 * @return Pointer to a `Prebuilt` instance, created from the files at `prefix`.
 * If no loading function can be found for the search algorithm at `prefix`, a `LoadPrebuiltNotFoundError` is thrown.
 */
template<typename Index_, typename Data_, typename Distance_>
Prebuilt<Index_, Data_, Distance_>* load_prebuilt_raw(const std::string& prefix) {
    const auto alg_path = prefix + "ALGORITHM";
    const auto algorithm = quick_load_as_string(alg_path);

    const auto& reg = load_prebuilt_registry<Index_, Data_, Distance_>(); 
    auto it = reg.find(algorithm);
    if (it == reg.end()) {
        throw LoadPrebuiltNotFoundError(algorithm, alg_path);
    }

    return (it->second)(prefix);
}

/**
 * Load a neighbor search index from disk into a `Prebuilt` object. 
 * This should be called with the same template parameters as the `Prebuilt` interface from which `save()` was called.
 *
 * @tparam Index_ Integer type for the observation indices.
 * @tparam Data_ Numeric type for the query data.
 * @tparam Distance_ Floating point type for the distances.
 *
 * @param prefix File path prefix for a prebuilt index that was saved to disk by `Prebuilt::save()`.
 *
 * @return Unique pointer to a `Prebuilt` instance, created from the files at `prefix`.
 * This uses the return value of `load_prebuilt_raw()`. 
 */
template<typename Index_, typename Data_, typename Distance_>
std::unique_ptr<Prebuilt<Index_, Data_, Distance_> > load_prebuilt_unique(const std::string& prefix) {
    return std::unique_ptr<Prebuilt<Index_, Data_, Distance_> >(load_prebuilt_raw<Index_, Data_, Distance_>(prefix));
}

/**
 * Load a neighbor search index from disk into a `Prebuilt` object. 
 * This should be called with the same template parameters as the `Prebuilt` interface from which `save()` was called.
 *
 * @tparam Index_ Integer type for the observation indices.
 * @tparam Data_ Numeric type for the query data.
 * @tparam Distance_ Floating point type for the distances.
 *
 * @param prefix File path prefix for a prebuilt index that was saved to disk by `Prebuilt::save()`.
 *
 * @return Shared pointer to a `Prebuilt` instance, created from the files at `prefix`.
 * This uses the return value of `load_prebuilt_raw()`. 
 */
template<typename Index_, typename Data_, typename Distance_>
std::shared_ptr<Prebuilt<Index_, Data_, Distance_> > load_prebuilt_shared(const std::string& prefix) {
    return std::shared_ptr<Prebuilt<Index_, Data_, Distance_> >(load_prebuilt_raw<Index_, Data_, Distance_>(prefix));
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
    knncolle::NumericType normalized;
};

/**
 * @param prefix Prefix of the file paths in which a prebuilt L2-normalized index was saved.
 * Files should have been generated by the `Prebuilt::save()` method of the L2-normalized `Prebuilt` subclass instance.
 *
 * @return Template types of the saved instance of a `Prebuilt` L2-normalized subclass.
 * This is typically used to choose template parameters for `load_l2normalized_prebuilt()`.
 */
inline L2NormalizedPrebuiltTypes load_l2normalized_prebuilt_types(const std::string& prefix) {
    L2NormalizedPrebuiltTypes config;
    quick_load(prefix + "normalized", &(config.normalized), 1);
    return config;
}

/**
 * Load an L2-normalized search index, i.e., a `knncolle::Prebuilt` created by `L2NormalizedBuilder`.
 * This is not provided in the registry by default as its depends on the application's set of the acceptable `Normalized_` types.
 * The `Normalized_` type in the saved index can be retrived by `load_l2normalized_prebuilt_types()`.
 *
 * @tparam Index_ Integer type for the indices.
 * @tparam Data_ Numeric type for the input and query data.
 * @tparam Distance_ Floating-point type for the distances.
 * @tparam Normalized_ Floating-point type for the L2-normalized data.
 *
 * @param prefix Prefix of the file paths in which a prebuilt L2-normalized index was saved.
 * Files should have been generated by the `Prebuilt::save()` method of the L2-normalized `Prebuilt` subclass instance.
 * 
 * @return Pointer to an L2-normalized `knncolle::Prebuilt` instance.
 */
template<typename Index_, typename Data_, typename Distance_, typename Normalized_>
Prebuilt<Index_, Data_, Distance_>* load_l2normalized_prebuilt(const std::string& prefix) {
    return new L2NormalizedPrebuilt<Index_, Data_, Distance_, Normalized_>(prefix);
}

}

#endif
