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
    auto bptr = bb.build_unique(mat);

    // Testing other types. 
    knncolle::SimpleMatrix<int, size_t, double> mat2(ndim, nobs, data.data());
    knncolle::KmknnBuilder<knncolle::EuclideanDistance, decltype(mat2), float> kb2;
    auto kptr2 = kb2.build_unique(mat2);

    std::vector<int> kres_i, ref_i;
    std::vector<double> kres_d, ref_d;
    auto bsptr = bptr->initialize();
    auto ksptr = kptr->initialize();
    std::vector<size_t> kres2_i;
    std::vector<float> kres2_d;
    auto ksptr2 = kptr2->initialize();

    for (int x = 0; x < nobs; ++x) {
        ksptr->search(x, k, &kres_i, &kres_d);
        bsptr->search(x, k, &ref_i, &ref_d);
        EXPECT_EQ(kres_i, ref_i);
        EXPECT_EQ(kres_d, ref_d);

        // Trying with some NULLs.
        ksptr->search(x, k, NULL, &kres_d);
        EXPECT_EQ(kres_d, ref_d);
        ksptr->search(x, k, &kres_i, NULL);
        EXPECT_EQ(kres_i, ref_i);

        ksptr2->search(x, k, &kres2_i, &kres2_d);
        EXPECT_EQ(kres_i.size(), kres2_i.size());
        for (size_t i = 0; i < kres_i.size(); ++i) {
            EXPECT_EQ(kres_i[i], kres2_i[i]);
            EXPECT_FLOAT_EQ(kres_d[i], kres2_d[i]);
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

    std::vector<int> kres_i, ref_i;
    std::vector<double> kres_d, ref_d;
    auto bsptr = bptr->initialize();
    auto ksptr = kptr->initialize();

    for (int x = 0; x < nobs; ++x) {
        ksptr->search(x, k, &kres_i, &kres_d);
        bsptr->search(x, k, &ref_i, &ref_d);
        EXPECT_EQ(kres_i, ref_i);
        EXPECT_EQ(kres_d, ref_d);
    }
}

TEST_P(KmknnTest, QueryEuclidean) {
    int k = std::get<1>(GetParam());    

    knncolle::SimpleMatrix mat(ndim, nobs, data.data());
    knncolle::KmknnBuilder<> kb;
    auto kptr = kb.build_unique(mat);
    knncolle::BruteforceBuilder<> bb;
    auto bptr = bb.build_unique(mat);

    std::vector<int> kres_i, ref_i;
    std::vector<double> kres_d, ref_d;
    auto bsptr = bptr->initialize();
    auto ksptr = kptr->initialize();

    std::mt19937_64 rng(ndim * 10 + nobs - k);
    std::vector<double> buffer(ndim);

    for (int x = 0; x < nobs; ++x) {
        fill_random(buffer.begin(), buffer.end(), rng);
        ksptr->search(buffer.data(), k, &kres_i, &kres_d);
        bsptr->search(buffer.data(), k, &ref_i, &ref_d);
        EXPECT_EQ(kres_i, ref_i);
        EXPECT_EQ(kres_d, ref_d);

        // Trying with some NULLs.
        ksptr->search(buffer.data(), k, NULL, &kres_d);
        EXPECT_EQ(kres_d, ref_d);
        ksptr->search(buffer.data(), k, &kres_i, NULL);
        EXPECT_EQ(kres_i, ref_i);
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

class KmknnDuplicateTest : public TestCore, public ::testing::TestWithParam<int> {
protected:
    void SetUp() {
        assemble({ 5, 3 });
    }
};

TEST_P(KmknnDuplicateTest, Basic) {
    // The duplicate testing checks that KMKNN handles zero-size clusters
    // correctly. With the default kmeans++ initialization, some of the
    // clusters will be empty if 'k' is larger than the number of unique
    // points; these should be filtered out during KmknnPrebuilt construction.

    int duplication = 10;
    std::vector<double> dup;
    for (int d = 0; d < duplication; ++d) {
        dup.insert(dup.end(), data.begin(), data.end());
    }

    knncolle::KmknnBuilder<> bb;
    int actual_nobs = nobs * duplication;
    auto bptr = bb.build_unique(knncolle::SimpleMatrix(ndim, actual_nobs, dup.data()));
    auto bsptr = bptr->initialize();
    std::vector<int> res_i;
    std::vector<double> res_d;

    int k = GetParam();
    for (int o = 0; o < actual_nobs; ++o) {
        bsptr->search(o, k, &res_i, &res_d);
        int full_set = std::min(k, actual_nobs - 1);
        EXPECT_EQ(res_i.size(), full_set);
        EXPECT_EQ(res_d.size(), full_set);

        int all_equal = std::min(k, duplication - 1);
        for (int i = 0; i < all_equal; ++i) {
            EXPECT_EQ(res_i[i] % nobs, o % nobs);
            EXPECT_EQ(res_d[i], 0);
        }

        for (int i = all_equal; i < full_set; ++i) {
            EXPECT_NE(res_i[i] % nobs, o % nobs);
            EXPECT_GT(res_d[i], 0);
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    Kmknn,
    KmknnDuplicateTest,
    ::testing::Values(3, 10, 20) // number of neighbors
);
