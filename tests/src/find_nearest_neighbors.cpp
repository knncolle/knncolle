#include <gtest/gtest.h>

#ifdef TEST_KNNCOLLE_CUSTOM_PARALLEL
#include <cmath>
#include <vector>
#include <thread>
#include <iostream>

template<class Function_>
void custom_parallelize(size_t n, size_t nthreads, Function_ f) {
    size_t jobs_per_worker = std::ceil(static_cast<double>(n) / nthreads);
    size_t start = 0;
    std::vector<std::thread> jobs;

    for (size_t w = 0; w < nthreads; ++w) {
        if (start >= n) {
            break;
        }
        size_t len = std::min(n - start, jobs_per_worker);
        jobs.emplace_back(f, start, len);
        start += len;
    }

    for (auto& job : jobs) {
        job.join();
    }
}

#define KNNCOLLE_CUSTOM_PARALLEL custom_parallelize
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

    auto base = knncolle::VptreeBuilder<>().build_unique(knncolle::SimpleMatrix(ndim, nobs, data.data()));
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

    knncolle::SimpleMatrix mat(ndim, nobs, data.data());
    auto base = knncolle::VptreeBuilder<>().build_unique(mat);
    auto ref = knncolle::find_nearest_neighbors<>(*base, k, 1);

    knncolle::SimpleMatrix<double, size_t, int> mat2(ndim, nobs, data.data());
    auto base2 = knncolle::VptreeBuilder<knncolle::EuclideanDistance, decltype(mat2), float>().build_unique(mat2);
    auto out2 = knncolle::find_nearest_neighbors(*base2, k, 1);

    EXPECT_EQ(out2.size(), nobs);
    for (int i = 0; i < nobs; ++i) {
        const auto& left = ref[i];
        const auto& right = out2[i];
        EXPECT_EQ(left.size(), std::min(k, nobs - 1));
        EXPECT_EQ(right.size(), left.size());

        for (size_t j = 0; j < left.size(); ++j) {
            EXPECT_EQ(left[j].first, right[j].first);
            EXPECT_FLOAT_EQ(left[j].second, right[j].second);
        }
    }
}

TEST_P(FindNearestNeighborsTest, IndexOnly) {
    int k = std::get<1>(GetParam());

    auto base = knncolle::VptreeBuilder<>().build_unique(knncolle::SimpleMatrix(ndim, nobs, data.data()));
    auto ref = knncolle::find_nearest_neighbors<>(*base, k, 1);
    auto out = knncolle::find_nearest_neighbors_index_only<>(*base, k, 1);

    EXPECT_EQ(out.size(), nobs);
    for (int i = 0; i < nobs; ++i) {
        const auto& left = ref[i];
        const auto& right = out[i];
        EXPECT_EQ(left.size(), std::min(k, nobs - 1));
        EXPECT_EQ(right.size(), left.size());

        for (size_t j = 0; j < left.size(); ++j) {
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
