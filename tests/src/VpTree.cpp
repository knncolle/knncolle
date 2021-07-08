#include <gtest/gtest.h>
#include "knncolle/VpTree/VpTree.hpp"
#include "knncolle/BruteForce/BruteForce.hpp"

#include <random>
#include <vector>

#include "TestCore.hpp"

class VpTreeTest : public TestCore<std::tuple<int, int, int> > {}; 

TEST_P(VpTreeTest, FindEuclidean) {
    auto param = GetParam();
    assemble(param);
    int k = std::get<2>(param);    

    knncolle::VpTreeEuclidean<> vp(nobs, ndim, data.data());
    knncolle::BruteForceEuclidean<> bf(nobs, ndim, data.data());

    for (size_t x = 0; x < nobs; ++x) {
        vp.find_nearest_neighbors(x, k);
        bf.find_nearest_neighbors(x, k);
        EXPECT_EQ(vp.neighbors(), bf.neighbors());
        EXPECT_EQ(vp.distances(), bf.distances());
    }
}

TEST_P(VpTreeTest, FindManhattan) {
    auto param = GetParam();
    assemble(param);
    int k = std::get<2>(param);    

    knncolle::VpTreeManhattan<> vp(nobs, ndim, data.data());
    knncolle::BruteForceManhattan<> bf(nobs, ndim, data.data());

    for (size_t x = 0; x < nobs; ++x) {
        vp.find_nearest_neighbors(x, k);
        bf.find_nearest_neighbors(x, k);
        EXPECT_EQ(vp.neighbors(), bf.neighbors());
        EXPECT_EQ(vp.distances(), bf.distances());
    }
}

INSTANTIATE_TEST_CASE_P(
    VpTree,
    VpTreeTest,
    ::testing::Combine(
        ::testing::Values(10, 500), // number of observations
        ::testing::Values(1, 20), // number of dimensions
        ::testing::Values(1, 10) // number of neighbors
    )
);

