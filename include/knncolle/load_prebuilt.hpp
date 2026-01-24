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
 * @return Reference to a global map of method names (see `Prebuilt::save()`) to loading functions.
 */
template<typename Index_, typename Data_, typename Distance_>
inline std::unordered_map<std::string, LoadPrebuiltFunction<Index_, Data_, Distance_> >& load_prebuilt_registry() {
    static std::unordered_map<std::string, LoadPrebuiltFunction<Index_, Data_, Distance_> > registry = default_prebuilt_registry<Index_, Data_, Distance_>();
    return registry;
}

/**
 * Load a neighbor search index from disk into a `Prebuilt` object.
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
    std::ifstream input(meth_path);
    std::string method( (std::istreambuf_iterator<char>(input)), (std::istreambuf_iterator<char>()) );

    const auto& reg = load_prebuilt_registry<Index_, Data_, Distance_>(); 
    auto it = reg.find(method);
    if (it == reg.end()) {
        throw std::runtime_error("cannot find load_prebuilt method for '" + method + "' at '" + meth_path + "'");
    }

    return (it->second)(prefix);
}

/**
 * Load a neighbor search index from disk into a `Prebuilt` object. 
 * This calls `load_prebuilt_raw()` internally.
 *
 * @tparam Index_ Integer type for the observation indices.
 * @tparam Data_ Numeric type for the query data.
 * @tparam Distance_ Floating point type for the distances.
 *
 * @param prefix File path prefix for a prebuilt index that was saved to disk by `Prebuilt::save()`.
 *
 * @return Unique pointer to a `Prebuilt` instance, created from the files at `prefix`.
 */
template<typename Index_, typename Data_, typename Distance_>
std::unique_ptr<Prebuilt<Index_, Data_, Distance_> > load_prebuilt_unique(const std::string& prefix) {
    return std::unique_ptr<Prebuilt<Index_, Data_, Distance_> >(load_prebuilt_raw<Index_, Data_, Distance_>(prefix));
}

/**
 * Load a neighbor search index from disk into a `Prebuilt` object. 
 * This calls `load_prebuilt_raw()` internally.
 *
 * @tparam Index_ Integer type for the observation indices.
 * @tparam Data_ Numeric type for the query data.
 * @tparam Distance_ Floating point type for the distances.
 *
 * @param prefix File path prefix for a prebuilt index that was saved to disk by `Prebuilt::save()`.
 *
 * @return Shared pointer to a `Prebuilt` instance, created from the files at `prefix`.
 */
template<typename Index_, typename Data_, typename Distance_>
std::shared_ptr<Prebuilt<Index_, Data_, Distance_> > load_prebuilt_shared(const std::string& prefix) {
    return std::shared_ptr<Prebuilt<Index_, Data_, Distance_> >(load_prebuilt_raw<Index_, Data_, Distance_>(prefix));
}

}

#endif
