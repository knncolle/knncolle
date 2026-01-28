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
 * @cond
 */
template<typename Index_, typename Data_, typename Distance_>
auto default_prebuilt_registry() {
    std::unordered_map<std::string, LoadPrebuiltFunction<Index_, Data_, Distance_> > registry;
    registry["knncolle::Bruteforce"] = [](const std::string& prefix) -> Prebuilt<Index_, Data_, Distance_>* {
        return new BruteforcePrebuilt<Index_, Data_, Distance_, DistanceMetric<Data_, Distance_> >(prefix);
    };
    registry["knncolle::Vptree"] = [](const std::string& prefix) -> Prebuilt<Index_, Data_, Distance_>* {
        return new VptreePrebuilt<Index_, Data_, Distance_, DistanceMetric<Data_, Distance_> >(prefix);
    };
    return registry;
}
/**
 * @endcond
 */

/**
 * @tparam Index_ Integer type for the observation indices.
 * @tparam Data_ Numeric type for the query data.
 * @tparam Distance_ Floating point type for the distances.
 *
 * @return Reference to a global map of method names (see `Prebuilt::save()`) to prebuilt loading functions.
 *
 * Note that no loading function is implemented by default for prebuilt indices created by `L2NormalizedBuilder()`.
 * This should be added separately with `l2normalized_save_name`, `load_l2normalized_prebuilt_types()`, and `load_l2normalized_prebuilt()`.
 */
template<typename Index_, typename Data_, typename Distance_>
inline std::unordered_map<std::string, LoadPrebuiltFunction<Index_, Data_, Distance_> >& load_prebuilt_registry() {
    static std::unordered_map<std::string, LoadPrebuiltFunction<Index_, Data_, Distance_> > registry = default_prebuilt_registry<Index_, Data_, Distance_>();
    return registry;
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
 * @return Pointer to a `Prebuilt` instance, created from the files at `prefix`.
 */
template<typename Index_, typename Data_, typename Distance_>
Prebuilt<Index_, Data_, Distance_>* load_prebuilt_raw(const std::string& prefix) {
    const auto meth_path = prefix + "ALGORITHM";
    auto method = quick_load_as_string(meth_path);

    const auto& reg = load_prebuilt_registry<Index_, Data_, Distance_>(); 
    auto it = reg.find(method);
    if (it == reg.end()) {
        throw std::runtime_error("cannot find load_prebuilt method for '" + method + "' at '" + meth_path + "'");
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
 * An L2-normalized index is typically saved by calling the `knncolle::Prebuilt::save()` method of the L2-normalized subclass instance.
 *
 * @return Template types of the saved instance of a `knncolle::Prebuilt` L2-normalized subclass.
 * This is typically used to choose template parameters for `load_l2normalized_prebuilt()`.
 */
inline L2NormalizedPrebuiltTypes load_l2normalized_prebuilt_types(const std::string& prefix) {
    L2NormalizedPrebuiltTypes config;
    quick_load(prefix + "normalized", &(config.normalized), 1);
    return config;
}

/**
 * Load an L2-normalized index, i.e., a `knncolle::Prebuilt` created by `L2NormalizedBuilder`.
 * This is not provided in the registry by default as its depends on the application's set of the acceptable `Normalized_` types.
 * The `Normalized_` type in the saved index can be retrived by `load_l2normalized_prebuilt_types()`.
 *
 * @tparam Index_ Integer type for the indices.
 * @tparam Data_ Numeric type for the input and query data.
 * @tparam Distance_ Floating-point type for the distances.
 * @tparam Normalized_ Floating-point type for the L2-normalized data.
 * 
 * @return Pointer to an L2-normalized `knncolle::Prebuilt` instance.
 */
template<typename Index_, typename Data_, typename Distance_, typename Normalized_>
Prebuilt<Index_, Data_, Distance_>* load_l2normalized_prebuilt(const std::string& prefix) {
    return new L2NormalizedPrebuilt<Index_, Data_, Distance_, Normalized_>(prefix);
}

}

#endif
