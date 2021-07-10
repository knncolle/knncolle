#include <gtest/gtest.h>
#include "knncolle/Annoy/Annoy.hpp"

#include <random>
#include <vector>

#include "TestCore.hpp"

class AnnoyTest : public TestCore<std::tuple<int, int, int> > {}; 

TEST_P(AnnoyTest, FindEuclidean) {
    auto param = GetParam();
    assemble(param);
    int k = std::get<2>(param);    

    knncolle::AnnoyEuclidean<> ann(ndim, nobs, data.data());

    std::vector<int> neighbors; 
    std::vector<double> distances;
    for (size_t x = 0; x < nobs; ++x) {
        ann.find_nearest_neighbors(x, k, &neighbors, &distances);
        sanity_checks(neighbors, distances, k, x);
    }
}

TEST_P(AnnoyTest, FindManhattan) {
    auto param = GetParam();
    assemble(param);
    int k = std::get<2>(param);    

    knncolle::AnnoyManhattan<> ann(ndim, nobs, data.data());

    std::vector<int> neighbors; 
    std::vector<double> distances;
    for (size_t x = 0; x < nobs; ++x) {
        ann.find_nearest_neighbors(x, k, &neighbors, &distances);
        sanity_checks(neighbors, distances, k, x);
    }
}

TEST_P(AnnoyTest, QueryEuclidean) {
    auto param = GetParam();
    assemble(param);
    int k = std::get<2>(param);

    knncolle::AnnoyEuclidean<> ann(ndim, nobs, data.data());

    std::vector<int> neighbors;
    std::vector<double> distances;
    for (size_t x = 0; x < nobs; ++x) {
        ann.find_nearest_neighbors(data.data() + x * ndim, k, &neighbors, &distances);
        sanity_checks(neighbors, distances, k);
    }
}

TEST_P(AnnoyTest, Options) {
    auto param = GetParam();
    assemble(param);
    int k = std::get<2>(param);    

    knncolle::AnnoyEuclidean<> vp(ndim, nobs, data.data());

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
    Annoy,
    AnnoyTest,
    ::testing::Combine(
        ::testing::Values(10, 500), // number of observations
        ::testing::Values(5, 20), // number of dimensions
        ::testing::Values(3, 10) // number of neighbors (one is greater than # observations, to test correct limiting)
    )
);

