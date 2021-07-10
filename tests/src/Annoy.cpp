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

    knncolle::AnnoyEuclidean<> annE(ndim, nobs, data.data());

    std::vector<int> neighbors; 
    std::vector<double> distances;
    for (size_t x = 0; x < nobs; ++x) {
        annE.find_nearest_neighbors(x, k, &neighbors, &distances);
        sanity_checks(neighbors, distances, k, x);
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
        sanity_checks(neighbors, distances, k, x);
    }
}


INSTANTIATE_TEST_CASE_P(
    Annoy,
    AnnoyTest,
    ::testing::Combine(
        ::testing::Values(10, 500), // number of observations
        ::testing::Values(5, 20), // number of dimensions
        ::testing::Values(3, 10) // number of neighbors
    )
);

