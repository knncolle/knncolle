#include <gtest/gtest.h>
#include "knncolle/BruteForce/BruteForce.hpp"

#include <vector>

#include "TestCore.hpp"

class BruteForceTest : public TestCore<std::tuple<int, int, int> > {}; 

TEST_P(BruteForceTest, FindEuclidean) {
    auto param = GetParam();
    assemble(param);
    int k = std::get<2>(param);    

    knncolle::BruteForceEuclidean<> bf(ndim, nobs, data.data());

    for (size_t x = 0; x < nobs; ++x) {
        auto results = bf.find_nearest_neighbors(x, k);
        sanity_checks(results, k, x);
    }
}

TEST_P(BruteForceTest, FindManhattan) {
    auto param = GetParam();
    assemble(param);
    int k = std::get<2>(param);    

    knncolle::BruteForceManhattan<> bf(ndim, nobs, data.data());

    for (size_t x = 0; x < nobs; ++x) {
        auto results = bf.find_nearest_neighbors(x, k);
        sanity_checks(results, k, x);
    }
}

TEST_P(BruteForceTest, QueryEuclidean) {
    auto param = GetParam();
    assemble(param);
    int k = std::get<2>(param);    

    knncolle::BruteForceEuclidean<> bf(ndim, nobs, data.data());

    for (size_t x = 0; x < nobs; ++x) {
        auto results1 = bf.find_nearest_neighbors(x, k);
        auto results2 = bf.find_nearest_neighbors(data.data() + x * ndim, k);

        EXPECT_EQ(results2[0].first, x);
        EXPECT_EQ(results2[0].second, 0);
        if (nobs > k) {
            results1.pop_back();
        }
        results2.erase(results2.begin());
        EXPECT_EQ(results1, results2);
    }
}

TEST_P(BruteForceTest, Getters) {
    auto param = GetParam();
    assemble(param);

    knncolle::BruteForceEuclidean<> bf(ndim, nobs, data.data());

    EXPECT_EQ(ndim, bf.ndim());

    EXPECT_EQ(nobs, bf.nobs());

    std::vector<double> buffer(ndim);
    auto ptr = bf.observation(2, buffer.data());
    compare_data(2, ptr);

    std::vector<double> buffer2 = bf.observation(5);
    compare_data(5, buffer2.data());
}

INSTANTIATE_TEST_CASE_P(
    BruteForce,
    BruteForceTest,
    ::testing::Combine(
        ::testing::Values(10, 500), // number of observations
        ::testing::Values(5, 20), // number of dimensions
        ::testing::Values(3, 10) // number of neighbors (one is greater than # observations, to test correct limiting)
    )
);

