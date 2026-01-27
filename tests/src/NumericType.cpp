#include <gtest/gtest.h>

#include "knncolle/NumericType.hpp"

#include <filesystem>
#include <string>
#include <vector>
#include <cstdint>
#include <cstddef>

TEST(GetNumericType, Basic) {
    // Don't do more specific tests because some compilers might have aliases
    // e.g., size_t might be a uint64_t and get reported as a UINT64_T.
    EXPECT_NE(knncolle::get_numeric_type<std::uint8_t>(), knncolle::NumericType::UNKNOWN);
    EXPECT_NE(knncolle::get_numeric_type<std::int8_t>(), knncolle::NumericType::UNKNOWN);

    EXPECT_NE(knncolle::get_numeric_type<std::uint16_t>(), knncolle::NumericType::UNKNOWN);
    EXPECT_NE(knncolle::get_numeric_type<std::int16_t>(), knncolle::NumericType::UNKNOWN);

    EXPECT_NE(knncolle::get_numeric_type<std::uint32_t>(), knncolle::NumericType::UNKNOWN);
    EXPECT_NE(knncolle::get_numeric_type<std::int32_t>(), knncolle::NumericType::UNKNOWN);

    EXPECT_NE(knncolle::get_numeric_type<std::uint64_t>(), knncolle::NumericType::UNKNOWN);
    EXPECT_NE(knncolle::get_numeric_type<std::int64_t>(), knncolle::NumericType::UNKNOWN);

    EXPECT_NE(knncolle::get_numeric_type<char>(), knncolle::NumericType::UNKNOWN);
    EXPECT_NE(knncolle::get_numeric_type<signed char>(), knncolle::NumericType::UNKNOWN);
    EXPECT_NE(knncolle::get_numeric_type<unsigned char>(), knncolle::NumericType::UNKNOWN);

    EXPECT_NE(knncolle::get_numeric_type<short>(), knncolle::NumericType::UNKNOWN);
    EXPECT_NE(knncolle::get_numeric_type<unsigned short>(), knncolle::NumericType::UNKNOWN);

    EXPECT_NE(knncolle::get_numeric_type<int>(), knncolle::NumericType::UNKNOWN);
    EXPECT_NE(knncolle::get_numeric_type<unsigned int>(), knncolle::NumericType::UNKNOWN);

    EXPECT_NE(knncolle::get_numeric_type<long>(), knncolle::NumericType::UNKNOWN);
    EXPECT_NE(knncolle::get_numeric_type<unsigned long>(), knncolle::NumericType::UNKNOWN);

    EXPECT_NE(knncolle::get_numeric_type<long long>(), knncolle::NumericType::UNKNOWN);
    EXPECT_NE(knncolle::get_numeric_type<unsigned long long>(), knncolle::NumericType::UNKNOWN);

    EXPECT_NE(knncolle::get_numeric_type<std::size_t>(), knncolle::NumericType::UNKNOWN);
    EXPECT_NE(knncolle::get_numeric_type<std::ptrdiff_t>(), knncolle::NumericType::UNKNOWN);

    EXPECT_NE(knncolle::get_numeric_type<float>(), knncolle::NumericType::UNKNOWN);
    EXPECT_NE(knncolle::get_numeric_type<double>(), knncolle::NumericType::UNKNOWN);

    EXPECT_EQ(knncolle::get_numeric_type<bool>(), knncolle::NumericType::UNKNOWN);
}
