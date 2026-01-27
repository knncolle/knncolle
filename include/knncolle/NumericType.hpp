#ifndef KNNCOLLE_NUMERIC_TYPE_HPP
#define KNNCOLLE_NUMERIC_TYPE_HPP

#include <cstddef>
#include <cstdint>
#include <type_traits>

/**
 * @file NumericType.hpp
 * @brief Preserve numeric types when saving prebuilt indices. 
 */

namespace knncolle {

/**
 * Standard numeric types, typically returned by `get_numeric_type()`.
 */
enum class NumericType : char {
    UINT8_T, INT8_T,
    UINT16_T, INT16_T,
    UINT32_T, INT32_T,
    UINT64_T, INT64_T,
    UNSIGNED_CHAR, SIGNED_CHAR, CHAR,
    UNSIGNED_SHORT, SHORT,
    UNSIGNED_INT, INT,
    UNSIGNED_LONG, LONG,
    UNSIGNED_LONG_LONG, LONG_LONG,
    SIZE_T, PTRDIFF_T,
    FLOAT, DOUBLE,
    UNKNOWN
};

/**
 * @tparam Type_ Some integer or floating-point type.
 * @return Identity of the numeric type.
 *
 * This function is intended for developers writing their own `Prebuilt::save()` methods,
 * where a subclass may have additional template types beyond those required by the `Prebuilt` template.
 * In such cases, developers can convert the type into a `NumericType` that can be saved to disk. 
 * The corresponding loader function can then read this type information to accurately reconstitute the original `Prebuilt` object.
 */
template<typename Type_>
NumericType get_numeric_type() {
#ifdef UINT8_MAX
    if constexpr(std::is_same<Type_, std::uint8_t>::value) {
        return NumericType::UINT8_T;
    }
#endif
#ifdef INT8_MAX
    if constexpr(std::is_same<Type_, std::int8_t>::value) {
        return NumericType::INT8_T;
    }
#endif

#ifdef UINT16_MAX
    if constexpr(std::is_same<Type_, std::uint16_t>::value) {
        return NumericType::UINT16_T;
    }
#endif
#ifdef INT16_MAX
    if constexpr(std::is_same<Type_, std::int16_t>::value) {
        return NumericType::INT16_T;
    }
#endif

#ifdef UINT32_MAX
    if constexpr(std::is_same<Type_, std::uint32_t>::value) {
        return NumericType::UINT32_T;
    }
#endif
#ifdef INT32_MAX
    if constexpr(std::is_same<Type_, std::int32_t>::value) {
        return NumericType::INT32_T;
    }
#endif

#ifdef UINT64_MAX
    if constexpr(std::is_same<Type_, std::uint64_t>::value) {
        return NumericType::UINT64_T;
    }
#endif
#ifdef INT64_MAX
    if constexpr(std::is_same<Type_, std::int64_t>::value) {
        return NumericType::INT64_T;
    }
#endif

    if constexpr(std::is_same<Type_, unsigned char>::value) {
        return NumericType::UNSIGNED_CHAR;
    }
    if constexpr(std::is_same<Type_, signed char>::value) {
        return NumericType::SIGNED_CHAR;
    }
    if constexpr(std::is_same<Type_, char>::value) {
        return NumericType::CHAR;
    }

    if constexpr(std::is_same<Type_, unsigned short>::value) {
        return NumericType::UNSIGNED_SHORT;
    }
    if constexpr(std::is_same<Type_, short>::value) {
        return NumericType::SHORT;
    }

    if constexpr(std::is_same<Type_, unsigned int>::value) {
        return NumericType::UNSIGNED_INT;
    }
    if constexpr(std::is_same<Type_, int>::value) {
        return NumericType::INT;
    }

    if constexpr(std::is_same<Type_, unsigned long>::value) {
        return NumericType::UNSIGNED_LONG;
    }
    if constexpr(std::is_same<Type_, long>::value) {
        return NumericType::LONG;
    }

    if constexpr(std::is_same<Type_, unsigned long long>::value) {
        return NumericType::UNSIGNED_LONG_LONG;
    }
    if constexpr(std::is_same<Type_, long long>::value) {
        return NumericType::LONG_LONG;
    }

    if constexpr(std::is_same<Type_, std::size_t>::value) {
        return NumericType::SIZE_T;
    }
    if constexpr(std::is_same<Type_, std::ptrdiff_t>::value) {
        return NumericType::PTRDIFF_T;
    }

    if constexpr(std::is_same<Type_, float>::value) {
        return NumericType::FLOAT;
    }
    if constexpr(std::is_same<Type_, double>::value) {
        return NumericType::DOUBLE;
    }

    return NumericType::UNKNOWN;
}

}

#endif
