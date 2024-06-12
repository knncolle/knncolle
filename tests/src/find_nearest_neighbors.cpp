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
        const auto& x_i = out[i].first;
        const auto& x_d = out[i].second;
        EXPECT_EQ(x_i.size(), std::min(k, nobs - 1));
        EXPECT_EQ(x_i.size(), x_d.size());

        for (const auto& y : x_i) {
            EXPECT_NE(y, i);
        }

        double last = 0;
        for (const auto& d : x_d) {
            EXPECT_TRUE(d > last);
            last = d;
        }
    }

    // Same results in parallel.
    auto par = knncolle::find_nearest_neighbors<>(*base, k, 3);
    ASSERT_EQ(par.size(), out.size());
    for (int i = 0; i < nobs; ++i) {
        EXPECT_EQ(out[i].first, par[i].first);
        EXPECT_EQ(out[i].second, par[i].second);
    }
}

TEST_P(FindNearestNeighborsTest, DifferentType) {
    int k = std::get<1>(GetParam());

    knncolle::SimpleMatrix mat(ndim, nobs, data.data());
    auto base = knncolle::VptreeBuilder<>().build_unique(mat);
    auto ref = knncolle::find_nearest_neighbors<>(*base, k, 1);

    knncolle::SimpleMatrix<int, size_t, double> mat2(ndim, nobs, data.data());
    auto base2 = knncolle::VptreeBuilder<knncolle::EuclideanDistance, decltype(mat2), float>().build_unique(mat2);
    auto out2 = knncolle::find_nearest_neighbors(*base2, k, 1);

    EXPECT_EQ(out2.size(), nobs);
    for (int i = 0; i < nobs; ++i) {
        const auto& left_i = ref[i].first;
        const auto& left_d = ref[i].second;
        const auto& right_i = out2[i].first;
        const auto& right_d = out2[i].second;

        EXPECT_EQ(right_i.size(), left_i.size());
        EXPECT_EQ(right_d.size(), left_i.size());
        for (size_t j = 0; j < left_i.size(); ++j) {
            EXPECT_EQ(left_i[j], right_i[j]);
            EXPECT_FLOAT_EQ(left_d[j], right_d[j]);
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
        const auto& left = ref[i].first;
        const auto& right = out[i];
        EXPECT_EQ(left, right);
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
