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

    std::vector<int> neighbors;
    std::vector<double> distances;
    for (size_t x = 0; x < nobs; ++x) {
        bf.find_nearest_neighbors(x, k, &neighbors, &distances);
        sanity_checks(neighbors, distances, k, x);
    }
}

TEST_P(BruteForceTest, FindManhattan) {
    auto param = GetParam();
    assemble(param);
    int k = std::get<2>(param);    

    knncolle::BruteForceManhattan<> vp(ndim, nobs, data.data());
    knncolle::BruteForceManhattan<> bf(ndim, nobs, data.data());

    std::vector<int> neighbors; 
    std::vector<double> distances;
    for (size_t x = 0; x < nobs; ++x) {
        vp.find_nearest_neighbors(x, k, &neighbors, &distances);
        sanity_checks(neighbors, distances, k, x);
    }
}

TEST_P(BruteForceTest, QueryEuclidean) {
    auto param = GetParam();
    assemble(param);
    int k = std::get<2>(param);    

    knncolle::BruteForceEuclidean<> bf(ndim, nobs, data.data());

    std::vector<int> neighbors1, neighbors2;
    std::vector<double> distances1, distances2;
    for (size_t x = 0; x < nobs; ++x) {
        bf.find_nearest_neighbors(x, k, &neighbors1, &distances1);
        bf.find_nearest_neighbors(data.data() + x * ndim, k, &neighbors2, &distances2);

        EXPECT_EQ(neighbors2[0], x);
        if (nobs > k) {
            neighbors1.pop_back();
        }
        neighbors2.erase(neighbors2.begin());
        EXPECT_EQ(neighbors1, neighbors2);

        EXPECT_EQ(distances2[0], 0);
        if (nobs > k) {
            distances1.pop_back();
        }
        distances2.erase(distances2.begin());
        EXPECT_EQ(distances1, distances2);
    }
}

TEST_P(BruteForceTest, Options) {
    auto param = GetParam();
    assemble(param);
    int k = std::get<2>(param);    

    knncolle::BruteForceEuclidean<> bf(ndim, nobs, data.data());

    std::vector<int> neighbors;
    std::vector<double> distances;
    bf.find_nearest_neighbors(0, k, NULL, &distances);
    EXPECT_TRUE(distances.size() >= 1);

    distances.clear();
    bf.find_nearest_neighbors(0, k, &neighbors, NULL);
    EXPECT_TRUE(neighbors.size() >= 1);
    EXPECT_TRUE(distances.size() == 0);
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

