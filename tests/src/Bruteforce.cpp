#include <gtest/gtest.h>
#include "knncolle/Bruteforce.hpp"

#include <vector>

#include "TestCore.hpp"

class BruteforceTest : public TestCore, public ::testing::TestWithParam<std::tuple<std::tuple<int, int>, int> > {
protected:
    void SetUp() {
        assemble(std::get<0>(GetParam()));
    }
};

TEST_P(BruteforceTest, FindEuclidean) {
    int k = std::get<1>(GetParam());    

    knncolle::BruteforceBuilder<> bb;
    auto bptr = bb.build_unique(knncolle::SimpleMatrix(ndim, nobs, data.data()));
    EXPECT_EQ(ndim, bptr->num_dimensions());
    EXPECT_EQ(nobs, bptr->num_observations());

    // Testing other types. 
    knncolle::SimpleMatrix<int, size_t, double> mat2(ndim, nobs, data.data());
    knncolle::BruteforceBuilder<knncolle::EuclideanDistance, decltype(mat2), float> bb2;
    auto bptr2 = bb2.build_unique(mat2);

    auto bsptr = bptr->initialize();
    std::vector<std::pair<int, double> > output;
    auto bsptr2 = bptr2->initialize();
    std::vector<std::pair<size_t, float> > output2;

    for (int x = 0; x < nobs; ++x) {
        bsptr->search(x, k, output);
        sanity_checks(output, k, x);

        bsptr2->search(x, k, output2);
        EXPECT_EQ(output.size(), output2.size());
        for (size_t i = 0; i < output.size(); ++i) {
            EXPECT_EQ(output2[i].first, output[i].first);
            EXPECT_FLOAT_EQ(output2[i].second, output[i].second);
        }
    }
}

TEST_P(BruteforceTest, FindManhattan) {
    int k = std::get<1>(GetParam());    

    knncolle::BruteforceBuilder<knncolle::ManhattanDistance> bb;
    auto bptr = bb.build_unique(knncolle::SimpleMatrix(ndim, nobs, data.data()));

    auto bsptr = bptr->initialize();
    std::vector<std::pair<int, double> > results;

    for (int x = 0; x < nobs; ++x) {
        bsptr->search(x, k, results);
        sanity_checks(results, k, x);
    }
}

TEST_P(BruteforceTest, QueryEuclidean) {
    int k = std::get<1>(GetParam());    

    knncolle::BruteforceBuilder<> bb;
    auto bptr = bb.build_shared(knncolle::SimpleMatrix(ndim, nobs, data.data())); // building a shared one for some variety.

    auto bsptr = bptr->initialize();
    std::vector<std::pair<int, double> > results1, results2;

    for (int x = 0; x < nobs; ++x) {
        bsptr->search(x, k, results1);
        EXPECT_EQ(results1.size(), std::min(k, nobs - 1));

        bsptr->search(data.data() + x * ndim, k + 1, results2);
        EXPECT_EQ(results2[0].first, x);
        EXPECT_EQ(results2[0].second, 0);

        results2.erase(results2.begin());
        EXPECT_EQ(results1, results2);
    }
}

INSTANTIATE_TEST_SUITE_P(
    Bruteforce,
    BruteforceTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(10, 500), // number of observations
            ::testing::Values(5, 20) // number of dimensions
        ),
        ::testing::Values(3, 10, 20) // number of neighbors (one is greater than # observations, to test correct limiting)
    )
);
