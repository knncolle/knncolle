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

    knncolle::VpTreeEuclidean<> vp(ndim, nobs, data.data());
    knncolle::BruteForceEuclidean<> bf(ndim, nobs, data.data());

    for (size_t x = 0; x < nobs; ++x) {
        auto vpres = vp.find_nearest_neighbors(x, k);
        auto bfres = bf.find_nearest_neighbors(x, k);
        EXPECT_EQ(vpres, bfres);
    }
}

TEST_P(VpTreeTest, FindManhattan) {
    auto param = GetParam();
    assemble(param);
    int k = std::get<2>(param);    

    knncolle::VpTreeManhattan<> vp(ndim, nobs, data.data());
    knncolle::BruteForceManhattan<> bf(ndim, nobs, data.data());

    for (size_t x = 0; x < nobs; ++x) {
        auto vpres = vp.find_nearest_neighbors(x, k);
        auto bfres = bf.find_nearest_neighbors(x, k);
        EXPECT_EQ(vpres, bfres);
    }
}

TEST_P(VpTreeTest, QueryEuclidean) {
    auto param = GetParam();
    assemble(param);
    int k = std::get<2>(param);    

    knncolle::VpTreeEuclidean<> vp(ndim, nobs, data.data());
    knncolle::BruteForceEuclidean<> bf(ndim, nobs, data.data());

    std::vector<int> bf_neighbors, vp_neighbors;
    std::vector<double> bf_distances, vp_distances;
    for (size_t x = 0; x < nobs; ++x) {
        auto vpres = vp.find_nearest_neighbors(data.data() + x * ndim, k);
        auto bfres = bf.find_nearest_neighbors(data.data() + x * ndim, k);
        EXPECT_EQ(vpres, bfres);
    }
}

TEST_P(VpTreeTest, Getters) {
    auto param = GetParam();
    assemble(param);

    knncolle::VpTreeEuclidean<> vp(ndim, nobs, data.data());

    EXPECT_EQ(ndim, vp.ndim());

    EXPECT_EQ(nobs, vp.nobs());

    std::vector<double> buffer(ndim);
    auto ptr = vp.observation(2, buffer.data());
    compare_data(2, ptr);

    std::vector<double> buffer2 = vp.observation(5);
    compare_data(5, buffer2.data());
}

INSTANTIATE_TEST_CASE_P(
    VpTree,
    VpTreeTest,
    ::testing::Combine(
        ::testing::Values(10, 500), // number of observations
        ::testing::Values(1, 20), // number of dimensions
        ::testing::Values(3, 10) // number of neighbors (one is greater than # observations, to test correct limiting)
    )
);

