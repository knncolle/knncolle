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

    std::vector<int> vres_i, ref_i;
    std::vector<double> vres_d, ref_d;
    auto bsptr = bptr->initialize();
    auto vsptr = vptr->initialize();
    std::vector<size_t> vres2_i;
    std::vector<float> vres2_d;
    auto vsptr2 = vptr2->initialize();

    for (int x = 0; x < nobs; ++x) {
        vsptr->search(x, k, &vres_i, &vres_d);
        bsptr->search(x, k, &ref_i, &ref_d);
        EXPECT_EQ(vres_i, ref_i);
        EXPECT_EQ(vres_d, ref_d);

        // Throwing in some NULLs.
        vsptr->search(x, k, NULL, &vres_d);
        EXPECT_EQ(vres_d, ref_d);
        vsptr->search(x, k, &vres_i, NULL);
        EXPECT_EQ(vres_i, ref_i);

        vsptr2->search(x, k, &vres2_i, &vres2_d);
        EXPECT_EQ(vres_i.size(), vres2_i.size());
        for (size_t i = 0; i < vres_i.size(); ++i) {
            EXPECT_EQ(vres_i[i], vres2_i[i]);
            EXPECT_FLOAT_EQ(vres_d[i], vres2_d[i]);
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

    std::vector<int> vres_i, ref_i;
    std::vector<double> vres_d, ref_d;
    auto bsptr = bptr->initialize();
    auto vsptr = vptr->initialize();

    for (int x = 0; x < nobs; ++x) {
        vsptr->search(x, k, &vres_i, &vres_d);
        bsptr->search(x, k, &ref_i, &ref_d);
        EXPECT_EQ(vres_i, ref_i);
        EXPECT_EQ(vres_d, ref_d);
    }
}

TEST_P(VptreeTest, QueryEuclidean) {
    int k = std::get<1>(GetParam());    

    knncolle::SimpleMatrix mat(ndim, nobs, data.data());
    knncolle::VptreeBuilder<> vb;
    auto vptr = vb.build_unique(mat);
    knncolle::BruteforceBuilder<> bb;
    auto bptr = bb.build_unique(mat);

    std::vector<int> vres_i, ref_i;
    std::vector<double> vres_d, ref_d;
    auto bsptr = bptr->initialize();
    auto vsptr = vptr->initialize();

    std::mt19937_64 rng(ndim * 10 + nobs - k);
    std::vector<double> buffer(ndim);

    for (int x = 0; x < nobs; ++x) {
        fill_random(buffer.begin(), buffer.end(), rng);
        vsptr->search(buffer.data(), k, &vres_i, &vres_d);
        bsptr->search(buffer.data(), k, &ref_i, &ref_d);
        EXPECT_EQ(vres_i, ref_i);
        EXPECT_EQ(vres_d, ref_d);

        // Throwing in some NULLs.
        vsptr->search(buffer.data(), k, NULL, &vres_d);
        EXPECT_EQ(vres_d, ref_d);
        vsptr->search(buffer.data(), k, &vres_i, NULL);
        EXPECT_EQ(vres_i, ref_i);
    }
}

TEST_P(VptreeTest, AllEuclidean) {
    int k = std::get<1>(GetParam());    

    knncolle::VptreeBuilder<> vb;
    auto vptr = vb.build_unique(knncolle::SimpleMatrix(ndim, nobs, data.data()));
    auto vsptr = vptr->initialize();
    std::vector<int> output_i, ref_i;
    std::vector<double> output_d, ref_d;

    EXPECT_TRUE(vsptr->can_search_all());

    for (int x = 0; x < nobs; ++x) {
        {
            vsptr->search(x, k, &ref_i, &ref_d);
            double new_threshold = ref_d.back() * 0.99;
            while (ref_d.size() && ref_d.back() > new_threshold) {
                ref_d.pop_back();
                ref_i.pop_back();
            }

            vsptr->search_all(x, new_threshold, &output_i, &output_d);
            EXPECT_EQ(output_i, ref_i); 
            EXPECT_EQ(output_d, ref_d);

            vsptr->search_all(x, new_threshold, NULL, &output_d);
            EXPECT_EQ(output_d, ref_d);
            vsptr->search_all(x, new_threshold, &output_i, NULL);
            EXPECT_EQ(output_i, ref_i);
        }

        {
            auto ptr = data.data() + x * ndim;
            vsptr->search(ptr, k, &ref_i, &ref_d);
            double new_threshold = ref_d.back() * 0.99;
            while (ref_d.size() && ref_d.back() > new_threshold) {
                ref_d.pop_back();
                ref_i.pop_back();
            }

            vsptr->search_all(ptr, new_threshold, &output_i, &output_d);
            EXPECT_EQ(output_i, ref_i);
            EXPECT_EQ(output_d, ref_d);

            vsptr->search_all(ptr, new_threshold, NULL, &output_d);
            EXPECT_EQ(output_d, ref_d);
            vsptr->search_all(ptr, new_threshold, &output_i, NULL);
            EXPECT_EQ(output_i, ref_i);
        }
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

class VptreeDuplicateTest : public TestCore, public ::testing::TestWithParam<int> {
protected:
    void SetUp() {
        assemble({ 5, 3 });
    }
};

TEST_P(VptreeDuplicateTest, Basic) {
    // Checking for correct elimination of self when reporting from a
    // NeighborQueue, while in the presence of many duplicates that could push
    // out 'self' from the results.

    int duplication = 10;
    std::vector<double> dup;
    for (int d = 0; d < duplication; ++d) {
        dup.insert(dup.end(), data.begin(), data.end());
    }

    knncolle::VptreeBuilder<> bb;
    int actual_nobs = nobs * duplication;
    auto bptr = bb.build_unique(knncolle::SimpleMatrix(ndim, actual_nobs, dup.data()));
    auto bsptr = bptr->initialize();
    std::vector<int> ires;
    std::vector<double> dres;

    int k = GetParam();
    for (int o = 0; o < actual_nobs; ++o) {
        bsptr->search(o, k, &ires, &dres);
        int full_set = std::min(k, actual_nobs - 1);
        EXPECT_EQ(ires.size(), full_set);
        EXPECT_EQ(dres.size(), full_set);

        int all_equal = std::min(k, duplication - 1);
        for (int i = 0; i < all_equal; ++i) {
            EXPECT_EQ(ires[i] % nobs, o % nobs);
            EXPECT_EQ(dres[i], 0);
        }

        for (int i = all_equal; i < full_set; ++i) {
            EXPECT_NE(ires[i] % nobs, o % nobs);
            EXPECT_GT(dres[i], 0);
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    Vptree,
    VptreeDuplicateTest,
    ::testing::Values(3, 10, 20) // number of neighbors
);

TEST(Vptree, Empty) {
    int ndim = 5;
    int nobs = 0;
    std::vector<double> data;

    knncolle::VptreeBuilder<> vb;
    auto vptr = vb.build_unique(knncolle::SimpleMatrix(ndim, nobs, data.data()));
    auto vsptr = vptr->initialize();
    std::vector<int> res_i(10);
    std::vector<double> res_d(10);

    std::vector<double> target(ndim);
    vsptr->search(target.data(), 0, &res_i, &res_d);
    EXPECT_TRUE(res_i.empty());
    EXPECT_TRUE(res_d.empty());
}
