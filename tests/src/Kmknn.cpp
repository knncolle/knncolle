#include <gtest/gtest.h>
#include "knncolle/Kmknn.hpp"
#include "knncolle/Bruteforce.hpp"

#include <vector>
#include <random>

#include "TestCore.hpp"

class KmknnTest : public TestCore, public ::testing::TestWithParam<std::tuple<std::tuple<int, int>, int> > {
protected:
    void SetUp() {
        assemble(std::get<0>(GetParam()));
    }
}; 

TEST_P(KmknnTest, FindEuclidean) {
    int k = std::get<1>(GetParam());

    knncolle::SimpleMatrix mat(ndim, nobs, data.data());
    knncolle::KmknnBuilder<> kb;
    auto kptr = kb.build_unique(mat);
    EXPECT_EQ(ndim, kptr->num_dimensions());
    EXPECT_EQ(nobs, kptr->num_observations());

    // Building a brute-force reference.
    knncolle::BruteforceBuilder<> bb;
    auto bptr = kb.build_unique(mat);

    // Testing other types. 
    knncolle::SimpleMatrix<double, size_t, int> mat2(ndim, nobs, data.data());
    knncolle::KmknnBuilder<knncolle::EuclideanDistance, decltype(mat2), float> kb2;
    auto kptr2 = kb2.build_unique(mat2);

    std::vector<std::pair<int, double> > kresults, bresults;
    std::vector<std::pair<size_t, float> > kresults2;
    for (int x = 0; x < nobs; ++x) {
        kptr->search(x, k, kresults);
        bptr->search(x, k, bresults);
        EXPECT_EQ(kresults, bresults);

        kptr2->search(x, k, kresults2);
        EXPECT_EQ(kresults.size(), kresults2.size());
        for (size_t i = 0; i < kresults.size(); ++i) {
            EXPECT_EQ(kresults[i].first, kresults2[i].first);
            EXPECT_FLOAT_EQ(kresults[i].second, kresults2[i].second);
        }
    }
}

TEST_P(KmknnTest, FindManhattan) {
    int k = std::get<1>(GetParam());    

    knncolle::SimpleMatrix mat(ndim, nobs, data.data());
    knncolle::BruteforceBuilder<knncolle::ManhattanDistance> bb;
    auto bptr = bb.build_unique(mat);

    // Injecting some more interesting options.
    knncolle::KmknnOptions<> opt;
    opt.initialize_algorithm.reset(new kmeans::InitializeRandom<>);
    opt.refine_algorithm.reset(new kmeans::RefineLloyd<>);
    knncolle::KmknnBuilder<knncolle::ManhattanDistance> kb(opt);
    auto kptr = kb.build_unique(mat);

    std::vector<std::pair<int, double> > kresults, bresults;
    for (int x = 0; x < nobs; ++x) {
        kptr->search(x, k, kresults);
        bptr->search(x, k, bresults);
        EXPECT_EQ(kresults, bresults);
    }
}

TEST_P(KmknnTest, QueryEuclidean) {
    int k = std::get<1>(GetParam());    

    knncolle::SimpleMatrix mat(ndim, nobs, data.data());
    knncolle::KmknnBuilder<> kb;
    auto kptr = kb.build_unique(mat);
    knncolle::BruteforceBuilder<> bb;
    auto bptr = bb.build_unique(mat);

    std::vector<std::pair<int, double> > kresults, bresults;
    std::mt19937_64 rng(ndim * 10 + nobs - k);
    std::vector<double> buffer(ndim);

    for (int x = 0; x < nobs; ++x) {
        fill_random(buffer.begin(), buffer.end(), rng);
        kptr->search(buffer.data(), k, kresults);
        bptr->search(buffer.data(), k, bresults);
        EXPECT_EQ(bresults, kresults);
    }
}

INSTANTIATE_TEST_SUITE_P(
    Kmknn,
    KmknnTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(10, 500), // number of observations
            ::testing::Values(5, 20) // number of dimensions
        ),
        ::testing::Values(3, 10, 20) // number of neighbors (one is greater than # observations, to test correct limiting)
    )
);
