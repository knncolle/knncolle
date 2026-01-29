#ifndef KNNCOLLE_PREBUILT_HPP
#define KNNCOLLE_PREBUILT_HPP

#include <memory>
#include <cstddef>
#include <filesystem>

#include "Searcher.hpp"

/**
 * @file Prebuilt.hpp
 *
 * @brief Interface for prebuilt nearest-neighbor indices.
 */

namespace knncolle {

/**
 * @brief Interface for prebuilt nearest-neighbor search indices.
 *
 * Instances of `Prebuilt` subclasses are typically constructed with `Builder::build_raw()`.
 * Note that a `Prebuilt` instance may outlive the `Builder` object used to generate it, so the former should not hold any references to the latter.
 *
 * @tparam Index_ Integer type for the observation indices.
 * @tparam Data_ Numeric type for the query data.
 * @tparam Distance_ Numeric type for the distances, usually floating-point.
 */
template<typename Index_, typename Data_, typename Distance_>
class Prebuilt {
public:
    /**
     * @cond
     */
    Prebuilt() = default;
    Prebuilt(const Prebuilt&) = default;
    Prebuilt(Prebuilt&&) = default;
    Prebuilt& operator=(const Prebuilt&) = default;
    Prebuilt& operator=(Prebuilt&&) = default;
    virtual ~Prebuilt() = default;
    /**
     * @endcond
     */

public:
    /**
     * @return Number of observations in the dataset to be searched.
     */
    virtual Index_ num_observations() const = 0;
    
    /**
     * @return Number of dimensions.
     */
    virtual std::size_t num_dimensions() const = 0;

public:
    /**
     * Create a `Searcher` for searching the index.
     * @return Pointer to a `Searcher` instance.
     */
    virtual std::unique_ptr<Searcher<Index_, Data_, Distance_> > initialize() const = 0;

    /**
     * Save the prebuilt index to disk, to be reloaded with `load_prebuilt_raw()` and friends.
     *
     * An implementation of this method should create an `ALGORITHM` file inside `dir` that contains the search algorithm's name.
     * This should be an ASCII file with no newlines, where the algorithm name should follow the `<library>::<algorithm>` format, e.g., `knncolle::Vptree`.
     * This will be used by `load_prebuilt_raw()` to determine the exact loader function to call. 
     *
     * Other than the `ALGORITHM` file, each implementation may create any number of additional files/directories of any format inside `dir`.
     * We recommend that the name of each file/directory immediately starts with an upper case letter and is in all-capitals.
     * This allows applications to add more custom files without the risk of conflicts, e.g., by naming them without an upper-case letter. 
     *
     * An implementation of this method is not required to use portable file formats.
     * `load_prebuilt_raw()` is only expected to work on the same system (i.e., architecture, compiler, compilation settings) that was used for the `save()` call.
     * Any additional portability is at the discretion of the implementation, e.g., it is common to assume IEEE floating-point and two's-complement integers.
     *
     * An implementation of this method is not required to create files that are readable by different versions of the implementation. 
     * Thus, the files created by this method are generally unsuitable for archival storage.
     * However, implementations are recommended to at least provide enough information to throw an exception if an incompatible version of `load_prebuilt_raw()` is used.
     *
     * If a subclass does not implement this method, an error is thrown by default.
     *
     * @param dir Path to a directory in which to save the index.
     * This directory should already exist.
     */
    virtual void save([[maybe_unused]] const std::filesystem::path& dir) const {
        throw std::runtime_error("saving is not supported");
    }

public:
    /**
     * @return Unqiue pointer to a `Searcher` subclass.
     *
     * Subclasses may override this method to return a pointer to a specific `Searcher` subclass.
     * This is used for devirtualization in other **knncolle** functions. 
     * If no override is provided, `initialize()` is called instead.
     */
    auto initialize_known() const {
        return initialize();
    }
};

}

#endif
