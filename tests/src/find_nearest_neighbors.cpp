#include <gtest/gtest.h>

#ifdef TEST_KNNCOLLE_CUSTOM_PARALLEL
#include <cmath>
#include <vector>
#include <thread>
#include <iostream>

template<class Function>
void parallelize(size_t n, Function f, size_t nthreads) {
    size_t jobs_per_worker = std::ceil(static_cast<double>(n) / nthreads);
    size_t start = 0;
    std::vector<std::thread> jobs;

    for (size_t w = 0; w < nthreads; ++w) {
        size_t end = std::min(n, start + jobs_per_worker);
        if (start >= end) {
            break;
        }
        jobs.emplace_back(f, start, end);
        start += jobs_per_worker;
    }

    for (auto& job : jobs) {
        job.join();
    }
}

#define KNNCOLLE_CUSTOM_PARALLEL parallelize
#endif

#include "knncolle/knncolle.hpp"

#include <vector>

#include "TestCore.hpp"

class FindNearestNeighborsTest : public TestCore<std::tuple<int, int, int> > {};

TEST_P(FindNearestNeighborsTest, Basic) {
    auto param = GetParam();
    assemble(param);
    int k = std::get<2>(param);    

    auto base = knncolle::VpTreeEuclidean<>(ndim, nobs, data.data());
    auto out = knncolle::find_nearest_neighbors<>(&base, k, 1);

    EXPECT_EQ(out.size(), nobs);
    for (size_t i = 0; i < nobs; ++i) {
        const auto& x = out[i];
        EXPECT_EQ(x.size(), k);

        double last = 0;
        for (const auto& y : x) {
            EXPECT_NE(y.first, i);
            EXPECT_TRUE(y.second > last);
            last = y.second;
        }
    }

    // Same results in parallel.
    auto par = knncolle::find_nearest_neighbors<>(&base, k, 3);
    ASSERT_EQ(par.size(), out.size());
    for (size_t i = 0; i < nobs; ++i) {
        EXPECT_EQ(out[i], par[i]);
    }
}

TEST_P(FindNearestNeighborsTest, DifferentType) {
    auto param = GetParam();
    assemble(param);
    int k = std::get<2>(param);    

    auto base = knncolle::VpTreeEuclidean<>(ndim, nobs, data.data());
    auto ref = knncolle::find_nearest_neighbors<>(&base, k, 1);
    auto out = knncolle::find_nearest_neighbors<size_t, float>(&base, k, 1);

    EXPECT_EQ(out.size(), nobs);
    for (size_t i = 0; i < nobs; ++i) {
        const auto& left = ref[i];
        const auto& right = out[i];
        EXPECT_EQ(left.size(), k);
        EXPECT_EQ(right.size(), k);
        for (size_t j = 0; j < k; ++j) {
            EXPECT_EQ(left[j].first, right[j].first);
            EXPECT_FLOAT_EQ(left[j].second, right[j].second);
        }
    }
}

TEST_P(FindNearestNeighborsTest, IndexOnly) {
    auto param = GetParam();
    assemble(param);
    int k = std::get<2>(param);    

    auto base = knncolle::VpTreeEuclidean<>(ndim, nobs, data.data());
    auto ref = knncolle::find_nearest_neighbors<>(&base, k, 1);
    auto out = knncolle::find_nearest_neighbors_index_only<>(&base, k, 1);

    EXPECT_EQ(out.size(), nobs);
    for (size_t i = 0; i < nobs; ++i) {
        const auto& left = ref[i];
        const auto& right = out[i];
        EXPECT_EQ(left.size(), k);
        EXPECT_EQ(right.size(), k);
        for (size_t j = 0; j < k; ++j) {
            EXPECT_EQ(left[j].first, right[j]);
        }
    }

    // Same results in parallel.
    auto par = knncolle::find_nearest_neighbors_index_only<>(&base, k, 3);
    ASSERT_EQ(par.size(), out.size());
    for (size_t i = 0; i < nobs; ++i) {
        EXPECT_EQ(out[i], par[i]);
    }
}

INSTANTIATE_TEST_CASE_P(
    FindNearestNeighbors,
    FindNearestNeighborsTest,
    ::testing::Combine(
        ::testing::Values(100), // number of observations
        ::testing::Values(5), // number of dimensions
        ::testing::Values(1, 10) // number of neighbors 
    )
);
