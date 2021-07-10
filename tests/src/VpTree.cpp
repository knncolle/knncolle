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

    std::vector<int> bf_neighbors, vp_neighbors;
    std::vector<double> bf_distances, vp_distances;
    for (size_t x = 0; x < nobs; ++x) {
        vp.find_nearest_neighbors(x, k, &vp_neighbors, &vp_distances);
        bf.find_nearest_neighbors(x, k, &bf_neighbors, &bf_distances);
        EXPECT_EQ(vp_neighbors, bf_neighbors);
        EXPECT_EQ(vp_distances, bf_distances);
    }
}

TEST_P(VpTreeTest, FindManhattan) {
    auto param = GetParam();
    assemble(param);
    int k = std::get<2>(param);    

    knncolle::VpTreeManhattan<> vp(ndim, nobs, data.data());
    knncolle::BruteForceManhattan<> bf(ndim, nobs, data.data());

    std::vector<int> bf_neighbors, vp_neighbors;
    std::vector<double> bf_distances, vp_distances;
    for (size_t x = 0; x < nobs; ++x) {
        vp.find_nearest_neighbors(x, k, &vp_neighbors, &vp_distances);
        bf.find_nearest_neighbors(x, k, &bf_neighbors, &bf_distances);
        EXPECT_EQ(vp_neighbors, bf_neighbors);
        EXPECT_EQ(vp_distances, bf_distances);
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
        bf.find_nearest_neighbors(data.data() + x * ndim, k, &vp_neighbors, &vp_distances);
        vp.find_nearest_neighbors(data.data() + x * ndim, k, &bf_neighbors, &bf_distances);
        EXPECT_EQ(vp_neighbors, bf_neighbors);
        EXPECT_EQ(vp_distances, bf_distances);
    }
}

TEST_P(VpTreeTest, Options) {
    auto param = GetParam();
    assemble(param);
    int k = std::get<2>(param);    

    knncolle::VpTreeEuclidean<> vp(ndim, nobs, data.data());

    std::vector<int> neighbors;
    std::vector<double> distances;
    vp.find_nearest_neighbors(0, k, NULL, &distances);
    EXPECT_TRUE(distances.size() >= 1);

    distances.clear();
    vp.find_nearest_neighbors(0, k, &neighbors, NULL);
    EXPECT_TRUE(neighbors.size() >= 1);
    EXPECT_TRUE(distances.size() == 0);
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

