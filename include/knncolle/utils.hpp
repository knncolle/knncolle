#ifndef KNNCOLLE_UTILS_HPP
#define KNNCOLLE_UTILS_HPP

#include <fstream>
#include <string>
#include <cstddef>
#include <type_traits>
#include <filesystem>

#include "sanisizer/sanisizer.hpp"

/**
 * @file utils.hpp
 * @brief Miscellaneous utilities for **knncolle**
 */

namespace knncolle {

/**
 * Saves an array to a binary file at `path`.
 * This is intended for developer use in implementations of `Prebuilt::save()`.
 *
 * @tparam Input_ Type of values to be saved.
 * @tparam Length_ Integer type of the length of the array.
 *
 * @param path File path to save to. 
 * Any directories in the path should already exist.
 * @param contents Pointer to an array of contents to be saved. 
 * @param length Length of the array, in terms of the number of elements (not the number of bytes). 
 * This should be non-negative.
 */
template<typename Input_, typename Length_>
void quick_save(const std::filesystem::path& path, const Input_* const contents, const Length_ length) {
    std::ofstream output(path, std::ofstream::binary);
    if (!output) {
        throw std::runtime_error("failed to open a binary file at '" + path.string() + "'");
    }

    output.exceptions(std::ofstream::failbit | std::ofstream::badbit);
    output.write(reinterpret_cast<const char*>(contents), sanisizer::product<std::streamsize>(sizeof(Input_), sanisizer::attest_gez(length)));
}

/**
 * Read an array from a binary file at `path`.
 * This is intended for developer use in `load_prebuilt()` functions. 
 *
 * @tparam Input_ Type of values to be read.
 * @tparam Length_ Integer type of the length of the array.
 *
 * @param path File path to read from. 
 * @param contents Pointer to an array in which to store the contents of `path`.
 * This should have at least `length` addressable elements.
 * @param length Number of elements (not bytes) to be read from `path`.
 * This should be non-negative.
 */
template<typename Input_, typename Length_>
void quick_load(const std::filesystem::path& path, Input_* const contents, const Length_ length) {
    std::ifstream input(path, std::ifstream::binary);
    if (!input) {
        throw std::runtime_error("failed to open a binary file at '" + path.string() + "'");
    }

    input.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    input.read(reinterpret_cast<char*>(contents), sanisizer::product<std::streamsize>(sizeof(Input_), sanisizer::attest_gez(length)));
}

/**
 * Read the contents of a binary file as a string.
 * This is intended for developer use in `load_prebuilt()` functions.
 *
 * @param path File path to read from.
 * @return Contents of `path` as a string.
 */
inline std::string quick_load_as_string(const std::filesystem::path& path) {
    std::ifstream input(path, std::ifstream::binary);
    if (!input) {
        throw std::runtime_error("failed to open a binary file at '" + path.string() + "'");
    }

    input.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    return std::string( (std::istreambuf_iterator<char>(input)), (std::istreambuf_iterator<char>()) );
}

/**
 * @cond
 */
template<typename Input_>
using I = std::remove_cv_t<std::remove_reference_t<Input_> >;
/**
 * @endcond
 */

}

#endif
