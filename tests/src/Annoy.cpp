#include <gtest/gtest.h>
#include "knncolle/Annoy/AnnoyBase.hpp"

#include <random>
#include <vector>

#include "TestCore.hpp"

class AnnoyTest : public TestCore<std::tuple<int, int, int> > {}; 

TEST_P(AnnoyTest, FindEuclidean) {
    auto param = GetParam();
    assemble(param);
    int k = std::get<2>(param);    

    knncolle::AnnoyEuclidean<> annE(ndim, nobs, data.data());

    std::vector<int> neighbors; 
    std::vector<double> distances;
    for (size_t x = 0; x < nobs; ++x) {
        annE.find_nearest_neighbors(x, k, &neighbors, &distances);
        EXPECT_EQ(neighbors.size(), distances.size());
        EXPECT_EQ(neighbors.size(), std::min(k, (int)nobs - 1));

        for (size_t i = 1; i < distances.size(); ++i) { // check for sortedness.
            EXPECT_TRUE(distances[i] >= distances[i-1]);
        }

        for (auto i : neighbors) { // self is not in there.
            EXPECT_TRUE(i != x);
        }
    }
}

TEST_P(AnnoyTest, FindManhattan) {
    auto param = GetParam();
    assemble(param);
    int k = std::get<2>(param);    

    knncolle::AnnoyManhattan<> annM(ndim, nobs, data.data());

    std::vector<int> neighbors; 
    std::vector<double> distances;
    for (size_t x = 0; x < nobs; ++x) {
        annM.find_nearest_neighbors(x, k, &neighbors, &distances);
        EXPECT_EQ(neighbors.size(), distances.size());
        EXPECT_EQ(neighbors.size(), std::min(k, (int)nobs - 1));

        for (size_t i = 1; i < distances.size(); ++i) { // check for sortedness.
            EXPECT_TRUE(distances[i] >= distances[i-1]);
        }

        for (auto i : neighbors) { // self is not in there.
            EXPECT_TRUE(i != x);
        }
    }
}


INSTANTIATE_TEST_CASE_P(
    Annoy,
    AnnoyTest,
    ::testing::Combine(
        ::testing::Values(10, 500), // number of observations
        ::testing::Values(1, 20), // number of dimensions
        ::testing::Values(1, 10) // number of neighbors
    )
);

