#include <gtest/gtest.h>

#include "knncolle/cap_k.hpp"

TEST(CapK, Basic) {
    EXPECT_EQ(knncolle::cap_k<int32_t>(10, 100), 10);
    EXPECT_EQ(knncolle::cap_k<int32_t>(10, 10), 9);
    EXPECT_EQ(knncolle::cap_k<int32_t>(10, 1), 0);
    EXPECT_EQ(knncolle::cap_k<int32_t>(10, 0), 0);

    EXPECT_EQ(knncolle::cap_k<uint32_t>(10, 100), 10);
    EXPECT_EQ(knncolle::cap_k<uint32_t>(10, 10), 9);
    EXPECT_EQ(knncolle::cap_k<uint32_t>(10, 1), 0);
    EXPECT_EQ(knncolle::cap_k<uint32_t>(10, 0), 0);
}

TEST(CapK, Query) {
    EXPECT_EQ(knncolle::cap_k_query<int32_t>(10, 100), 10);
    EXPECT_EQ(knncolle::cap_k_query<int32_t>(10, 10), 10);
    EXPECT_EQ(knncolle::cap_k_query<int32_t>(10, 1), 1);
    EXPECT_EQ(knncolle::cap_k_query<int32_t>(10, 0), 0);

    EXPECT_EQ(knncolle::cap_k_query<uint32_t>(10, 100), 10);
    EXPECT_EQ(knncolle::cap_k_query<uint32_t>(10, 10), 10);
    EXPECT_EQ(knncolle::cap_k_query<uint32_t>(10, 1), 1);
    EXPECT_EQ(knncolle::cap_k_query<uint32_t>(10, 0), 0);
}
