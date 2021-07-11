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

    for (size_t x = 0; x < nobs; ++x) {
        auto res = ann.find_nearest_neighbors(x, k);
        sanity_checks(res, k, x);
    }
}

TEST_P(AnnoyTest, FindManhattan) {
    auto param = GetParam();
    assemble(param);
    int k = std::get<2>(param);    

    knncolle::AnnoyManhattan<> ann(ndim, nobs, data.data());

    for (size_t x = 0; x < nobs; ++x) {
        auto res = ann.find_nearest_neighbors(x, k);
        sanity_checks(res, k, x);
    }
}

TEST_P(AnnoyTest, QueryEuclidean) {
    auto param = GetParam();
    assemble(param);
    int k = std::get<2>(param);

    knncolle::AnnoyEuclidean<> ann(ndim, nobs, data.data());

    for (size_t x = 0; x < nobs; ++x) {
        auto res = ann.find_nearest_neighbors(data.data() + x * ndim, k);
        sanity_checks(res, k);
    }
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

