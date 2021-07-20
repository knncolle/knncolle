#include <gtest/gtest.h>
#include "knncolle/knncolle.hpp"

#include <vector>

#include "TestCore.hpp"

class DispatchTest : public TestCore<std::tuple<int, int, int> > {};

TEST_P(DispatchTest, FindEuclidean) {
    auto param = GetParam();
    assemble(param);
    int k = std::get<2>(param);    

    knncolle::Dispatch<> dispatch;
    
    auto bfptr = dispatch.build(ndim, nobs, data.data(), knncolle::BRUTEFORCE);
    auto bfres = bfptr->find_nearest_neighbors(0, k);

    auto kmptr = dispatch.build(ndim, nobs, data.data(), knncolle::KMKNN);
    auto kmres = kmptr->find_nearest_neighbors(0, k);
    EXPECT_EQ(bfres, kmres);

    auto vpptr = dispatch.build(ndim, nobs, data.data(), knncolle::VPTREE);
    auto vpres = vpptr->find_nearest_neighbors(0, k);
    EXPECT_EQ(bfres, vpres);

    auto annptr = dispatch.build(ndim, nobs, data.data(), knncolle::ANNOY);
    auto annres = annptr->find_nearest_neighbors(0, k);
    EXPECT_EQ(bfres.size(), annres.size());

    auto nswptr = dispatch.build(ndim, nobs, data.data(), knncolle::HNSW);
    auto nswres = nswptr->find_nearest_neighbors(0, k);
    EXPECT_EQ(bfres.size(), nswres.size());
}

TEST_P(DispatchTest, FindManhattan) {
    auto param = GetParam();
    assemble(param);
    int k = std::get<2>(param);    

    knncolle::Dispatch<> dispatch;
    dispatch.distance_type = knncolle::MANHATTAN;

    auto bfptr = dispatch.build(ndim, nobs, data.data(), knncolle::BRUTEFORCE);
    auto bfres = bfptr->find_nearest_neighbors(1, k);

    auto kmptr = dispatch.build(ndim, nobs, data.data(), knncolle::KMKNN);
    auto kmres = kmptr->find_nearest_neighbors(1, k);
    EXPECT_EQ(bfres, kmres);

    auto vpptr = dispatch.build(ndim, nobs, data.data(), knncolle::VPTREE);
    auto vpres = vpptr->find_nearest_neighbors(1, k);
    EXPECT_EQ(bfres, vpres);

    auto annptr = dispatch.build(ndim, nobs, data.data(), knncolle::ANNOY);
    auto annres = annptr->find_nearest_neighbors(1, k);
    EXPECT_EQ(bfres.size(), annres.size());

    auto nswptr = dispatch.build(ndim, nobs, data.data(), knncolle::HNSW);
    auto nswres = nswptr->find_nearest_neighbors(1, k);
    EXPECT_EQ(bfres.size(), nswres.size());
}

INSTANTIATE_TEST_CASE_P(
    Dispatch,
    DispatchTest,
    ::testing::Combine(
        ::testing::Values(100), // number of observations
        ::testing::Values(5), // number of dimensions
        ::testing::Values(10) // number of neighbors 
    )
);
