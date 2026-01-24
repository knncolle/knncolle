#ifndef KNNCOLLE_UTILS_HPP
#define KNNCOLLE_UTILS_HPP

#include <fstream>
#include <string>
#include <cstddef>
#include <type_traits>

namespace knncolle {

template<typename Input_, typename Size_>
void quick_save(const std::string& path, const Input_* contents, const Size_ length) {
    std::ofstream output(path, std::ofstream::binary);
    output.exceptions(std::ofstream::failbit | std::ofstream::badbit);
    output.write(reinterpret_cast<const char*>(contents), sizeof(Input_) * length);
}

template<typename Input_, typename Size_>
void quick_load(const std::string& path, Input_* const contents, const Size_ length) {
    std::ifstream input(path, std::ifstream::binary);
    input.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    input.read(reinterpret_cast<char*>(contents), sizeof(Input_) * length);
}

template<typename Input_>
using I = std::remove_cv_t<std::remove_reference_t<Input_> >;

}

#endif
