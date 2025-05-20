#include <gtest/gtest.h>

#ifdef TEST_CUSTOM_PARALLEL
#include "subpar/subpar.hpp"
#define KNNCOLLE_CUSTOM_PARALLEL(nw, nt, fun) ::subpar::parallelize_range(nw, nt, std::move(fun));
#endif

#include "knncolle/knncolle.hpp"

#include <vector>

#include "TestCore.hpp"

class FindNearestNeighborsTest : public TestCore, public ::testing::TestWithParam<std::tuple<std::tuple<int, int>, int> > {
protected:
    void SetUp() {
        assemble(std::get<0>(GetParam()));
    }
};

TEST_P(FindNearestNeighborsTest, Basic) {
    int k = std::get<1>(GetParam());
    auto eucdist = std::make_shared<knncolle::EuclideanDistance<double, double> >();

    knncolle::VptreeBuilder<int, double, double> vb(eucdist);
    auto base = vb.build_unique(knncolle::SimpleMatrix<int, double>(ndim, nobs, data.data()));
    auto out = knncolle::find_nearest_neighbors<>(*base, k, 1);

    EXPECT_EQ(out.size(), nobs);
    for (int i = 0; i < nobs; ++i) {
        const auto& x = out[i];
        EXPECT_EQ(x.size(), std::min(k, nobs - 1));

        double last = 0;
        for (const auto& y : x) {
            EXPECT_NE(y.first, i);
            EXPECT_TRUE(y.second > last);
            last = y.second;
        }
    }

    // Same results in parallel.
    auto par = knncolle::find_nearest_neighbors<>(*base, k, 3);
    ASSERT_EQ(par.size(), out.size());
    for (int i = 0; i < nobs; ++i) {
        EXPECT_EQ(out[i], par[i]);
    }
}

TEST_P(FindNearestNeighborsTest, DifferentType) {
    int k = std::get<1>(GetParam());
    auto eucdist = std::make_shared<knncolle::EuclideanDistance<double, double> >();

    knncolle::VptreeBuilder<int, double, double> vb(eucdist);
    auto base = vb.build_unique(knncolle::SimpleMatrix<int, double>(ndim, nobs, data.data()));
    auto ref = knncolle::find_nearest_neighbors<>(*base, k, 1);

    knncolle::VptreeBuilder<size_t, double, double> vb2(eucdist);
    auto base2 = vb2.build_unique(knncolle::SimpleMatrix<size_t, double>(ndim, nobs, data.data()));
    auto out2 = knncolle::find_nearest_neighbors(*base2, k, 1);

    EXPECT_EQ(out2.size(), nobs);
    for (int i = 0; i < nobs; ++i) {
        const auto& left = ref[i];
        const auto& right = out2[i];

        EXPECT_EQ(right.size(), left.size());
        for (size_t j = 0; j < left.size(); ++j) {
            EXPECT_EQ(left[j].first, right[j].first);
            EXPECT_FLOAT_EQ(left[j].second, right[j].second);
        }
    }
}

TEST_P(FindNearestNeighborsTest, IndexOnly) {
    int k = std::get<1>(GetParam());
    auto eucdist = std::make_shared<knncolle::EuclideanDistance<double, double> >();

    knncolle::VptreeBuilder<int, double, double> vb(eucdist);
    auto base = vb.build_unique(knncolle::SimpleMatrix<int, double>(ndim, nobs, data.data()));
    auto ref = knncolle::find_nearest_neighbors<>(*base, k, 1);
    auto out = knncolle::find_nearest_neighbors_index_only<>(*base, k, 1);

    EXPECT_EQ(out.size(), nobs);
    for (int i = 0; i < nobs; ++i) {
        const auto& left = ref[i];
        const auto& right = out[i];
        EXPECT_EQ(left.size(), right.size());
        for (int j = 0, end = left.size(); j < end; ++j) {
            EXPECT_EQ(left[j].first, right[j]);
        }
    }

    // Same results in parallel.
    auto par = knncolle::find_nearest_neighbors_index_only<>(*base, k, 3);
    ASSERT_EQ(par.size(), out.size());
    for (int i = 0; i < nobs; ++i) {
        EXPECT_EQ(out[i], par[i]);
    }
}

INSTANTIATE_TEST_SUITE_P(
    FindNearestNeighbors,
    FindNearestNeighborsTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(5, 100), // number of observations
            ::testing::Values(5, 20) // number of dimensions
        ),
        ::testing::Values(1, 10) // number of neighbors 
    )
);

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
