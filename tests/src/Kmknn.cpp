#include <gtest/gtest.h>
#include "knncolle/Kmknn/Kmknn.hpp"
#include "knncolle/BruteForce/BruteForce.hpp"

#include <random>
#include <vector>

#include "TestCore.hpp"

class KmknnTest : public TestCore<std::tuple<int, int, int> > {}; 

TEST_P(KmknnTest, FindEuclidean) {
    auto param = GetParam();
    assemble(param);
    int k = std::get<2>(param);    

    knncolle::KmknnEuclidean<> km(ndim, nobs, data.data());
    knncolle::BruteForceEuclidean<> bf(ndim, nobs, data.data());

    for (size_t x = 0; x < nobs; ++x) {
        auto kmres = km.find_nearest_neighbors(x, k);
        auto bfres = bf.find_nearest_neighbors(x, k);
        EXPECT_EQ(kmres, bfres);
    }

    // Testing the float inputs.
    std::vector<float> fdata(data.begin(), data.end());
    knncolle::KmknnEuclidean<int, float> fkm(ndim, nobs, fdata.data());

    for (size_t x = 0; x < nobs; ++x) {
        auto kmres = km.find_nearest_neighbors(x, k);
        auto fkmres = fkm.find_nearest_neighbors(x, k);
        EXPECT_EQ(kmres.size(), fkmres.size());

        for (size_t j = 0; j < kmres.size(); ++j) {
            EXPECT_EQ(kmres[j].first, fkmres[j].first);
        }
    }
}

TEST_P(KmknnTest, FindManhattan) {
    auto param = GetParam();
    assemble(param);
    int k = std::get<2>(param);    

    knncolle::KmknnManhattan<> km(ndim, nobs, data.data());
    knncolle::BruteForceManhattan<> bf(ndim, nobs, data.data());

    for (size_t x = 0; x < nobs; ++x) {
        auto kmres = km.find_nearest_neighbors(x, k);
        auto bfres = bf.find_nearest_neighbors(x, k);
        EXPECT_EQ(kmres, bfres);
    }
}

TEST_P(KmknnTest, QueryEuclidean) {
    auto param = GetParam();
    assemble(param);
    int k = std::get<2>(param);    

    knncolle::KmknnEuclidean<> km(ndim, nobs, data.data());
    knncolle::BruteForceEuclidean<> bf(ndim, nobs, data.data());

    std::vector<int> bf_neighbors, km_neighbors;
    std::vector<double> bf_distances, km_distances;
    for (size_t x = 0; x < nobs; ++x) {
        auto kmres = km.find_nearest_neighbors(data.data() + x * ndim, k);
        auto bfres = bf.find_nearest_neighbors(data.data() + x * ndim, k);
        EXPECT_EQ(kmres, bfres);
    }
}

TEST_P(KmknnTest, Getters) {
    auto param = GetParam();
    assemble(param);

    knncolle::KmknnEuclidean<> km(ndim, nobs, data.data());

    EXPECT_EQ(ndim, km.ndim());

    EXPECT_EQ(nobs, km.nobs());

    std::vector<double> buffer(ndim);
    auto ptr = km.observation(2, buffer.data());
    compare_data(2, ptr);

    std::vector<double> buffer2 = km.observation(5);
    compare_data(5, buffer2.data());
}

INSTANTIATE_TEST_CASE_P(
    Kmknn,
    KmknnTest,
    ::testing::Combine(
        ::testing::Values(10, 500), // number of observations
        ::testing::Values(1, 20), // number of dimensions
        ::testing::Values(3, 10) // number of neighbors (one is greater than # observations, to test correct limiting)
    )
);

