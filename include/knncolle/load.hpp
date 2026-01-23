#ifndef KNNCOLLE_LOAD_HPP
#define KNNCOLLE_LOAD_HPP

#include <string>
#include <unordered_map>
#include <functional>
#include <fstream>
#include <memory>

#include "Prebuilt.hpp"
#include "Bruteforce.hpp"
#include "Vptree.hpp"

namespace knncolle {

/**
 * Loading function, that accepts a file path prefix (see `Prebuilt::save()`) and returns a pointer to a `Prebuilt` instance.
 *
 * @tparam Index_ Integer type for the observation indices.
 * @tparam Data_ Numeric type for the query data.
 * @tparam Distance_ Floating point type for the distances.
 */
template<typename Index_, typename Data_, typename Distance_>
using LoadFunction = std::function<Prebuilt<Index_, Data_, Distance_>* (const std::string&)>;

/**
 * @tparam Index_ Integer type for the observation indices.
 * @tparam Data_ Numeric type for the query data.
 * @tparam Distance_ Floating point type for the distances.
 *
 * @return Reference to a global map of method names (see `Prebuilt::save()`) to loading functions.
 */
template<typename Index_, typename Data_, typename Distance_>
inline std::unordered_map<std::string, LoadFunction<Index_, Data_, Distance_> >& get_load_registry() {
    static std::unordered_map<std::string, LoadFunction<Index_, Data_, Distance_> > registry;
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
Prebuilt<Index_, Data_, Distance_>* load_raw(const std::string& prefix) {
    const auto meth_path = prefix + "METHOD";
    std::ifstream input(meth_path);
    std::string method( (std::istreambuf_iterator<char>(input)), (std::istreambuf_iterator<char>()) );

    const auto& reg = get_load_registry<Index_, Data_, Distance_>(); 
    auto it = reg.find(method);
    if (it == reg.end()) {
        throw std::runtime_error("cannot find load method for '" + method + "' at '" + meth_path + "'");
    }

    return (it->second)(prefix);
}

/**
 * Load a neighbor search index from disk into a `Prebuilt` object. 
 * This calls `load_raw()` internally.
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
std::unique_ptr<Prebuilt<Index_, Data_, Distance_> > load_unique(const std::string& prefix) {
    return std::unique_ptr<Prebuilt<Index_, Data_, Distance_> >(load_raw<Index_, Data_, Distance_>(prefix));
}

/**
 * Load a neighbor search index from disk into a `Prebuilt` object. 
 * This calls `load_raw()` internally.
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
std::unique_ptr<Prebuilt<Index_, Data_, Distance_> > load_shared(const std::string& prefix) {
    return std::shared_ptr<Prebuilt<Index_, Data_, Distance_> >(load_raw<Index_, Data_, Distance_>(prefix));
}

}

#endif
