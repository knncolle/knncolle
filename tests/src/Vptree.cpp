#include <gtest/gtest.h>
#include "knncolle/Vptree.hpp"
#include "knncolle/Bruteforce.hpp"

#include <vector>
#include <random>

#include "TestCore.hpp"

class VptreeTest : public TestCore, public ::testing::TestWithParam<std::tuple<std::tuple<int, int>, int> > {
protected:
    void SetUp() {
        assemble(std::get<0>(GetParam()));
    }
}; 

TEST_P(VptreeTest, FindEuclidean) {
    int k = std::get<1>(GetParam());

    knncolle::SimpleMatrix mat(ndim, nobs, data.data());
    knncolle::VptreeBuilder<> vb;
    auto vptr = vb.build_unique(mat);
    EXPECT_EQ(ndim, vptr->num_dimensions());
    EXPECT_EQ(nobs, vptr->num_observations());

    // Building a brute-force reference.
    knncolle::BruteforceBuilder<> bb;
    auto bptr = bb.build_unique(mat);

    // Testing other types. 
    knncolle::SimpleMatrix<int, size_t, double> mat2(ndim, nobs, data.data());
    knncolle::VptreeBuilder<knncolle::EuclideanDistance, decltype(mat2), float> vb2;
    auto vptr2 = vb2.build_unique(mat2);

    std::vector<std::pair<int, double> > vresults, bresults;
    auto bsptr = bptr->initialize();
    auto vsptr = vptr->initialize();
    std::vector<std::pair<size_t, float> > vresults2;
    auto vsptr2 = vptr2->initialize();

    for (int x = 0; x < nobs; ++x) {
        vsptr->search(x, k, vresults);
        bsptr->search(x, k, bresults);
        EXPECT_EQ(vresults, bresults);

        vsptr2->search(x, k, vresults2);
        EXPECT_EQ(vresults.size(), vresults2.size());
        for (size_t i = 0; i < vresults.size(); ++i) {
            EXPECT_EQ(vresults[i].first, vresults2[i].first);
            EXPECT_FLOAT_EQ(vresults[i].second, vresults2[i].second);
        }
    }
}

TEST_P(VptreeTest, FindManhattan) {
    int k = std::get<1>(GetParam());    

    knncolle::SimpleMatrix mat(ndim, nobs, data.data());
    knncolle::BruteforceBuilder<knncolle::ManhattanDistance> bb;
    auto bptr = bb.build_unique(mat);

    // Injecting some more interesting options.
    knncolle::VptreeBuilder<knncolle::ManhattanDistance> vb;
    auto vptr = vb.build_unique(mat);

    std::vector<std::pair<int, double> > vresults, bresults;
    auto bsptr = bptr->initialize();
    auto vsptr = vptr->initialize();

    for (int x = 0; x < nobs; ++x) {
        vsptr->search(x, k, vresults);
        bsptr->search(x, k, bresults);
        EXPECT_EQ(vresults, bresults);
    }
}

TEST_P(VptreeTest, QueryEuclidean) {
    int k = std::get<1>(GetParam());    

    knncolle::SimpleMatrix mat(ndim, nobs, data.data());
    knncolle::VptreeBuilder<> vb;
    auto vptr = vb.build_unique(mat);
    knncolle::BruteforceBuilder<> bb;
    auto bptr = bb.build_unique(mat);

    std::vector<std::pair<int, double> > vresults, bresults;
    auto bsptr = bptr->initialize();
    auto vsptr = vptr->initialize();

    std::mt19937_64 rng(ndim * 10 + nobs - k);
    std::vector<double> buffer(ndim);

    for (int x = 0; x < nobs; ++x) {
        fill_random(buffer.begin(), buffer.end(), rng);
        vsptr->search(buffer.data(), k, vresults);
        bsptr->search(buffer.data(), k, bresults);
        EXPECT_EQ(bresults, vresults);
    }
}

INSTANTIATE_TEST_SUITE_P(
    Vptree,
    VptreeTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(10, 500), // number of observations
            ::testing::Values(5, 20) // number of dimensions
        ),
        ::testing::Values(3, 10, 20) // number of neighbors (one is greater than # observations, to test correct limiting)
    )
);
