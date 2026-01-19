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
    auto eucdist = std::make_shared<knncolle::EuclideanDistance<double, double> >();

    knncolle::SimpleMatrix<int, double> mat(ndim, nobs, data.data());
    knncolle::VptreeBuilder<int, double, double> vb(eucdist);
    auto vptr = vb.build_unique(mat);
    EXPECT_EQ(ndim, vptr->num_dimensions());
    EXPECT_EQ(nobs, vptr->num_observations());

    // Building a brute-force reference.
    knncolle::BruteforceBuilder<int, double, double> bb(eucdist);
    auto bptr = bb.build_unique(mat);

    // Testing other types. 
    knncolle::SimpleMatrix<size_t, double> mat2(ndim, nobs, data.data());
    knncolle::VptreeBuilder<size_t, double, float> vb2(std::make_shared<knncolle::EuclideanDistance<double, float> >());
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
    auto mandist = std::make_shared<knncolle::ManhattanDistance<double, double> >();

    knncolle::SimpleMatrix<int, double> mat(ndim, nobs, data.data());
    knncolle::BruteforceBuilder<int, double, double> bb(mandist);
    auto bptr = bb.build_unique(mat);
    knncolle::VptreeBuilder<int, double, double> vb(mandist);
    auto vptr = vb.build_shared(mat); // trying the shared method, for some variety.

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
    auto eucdist = std::make_shared<knncolle::EuclideanDistance<double, double> >();

    knncolle::SimpleMatrix<int, double> mat(ndim, nobs, data.data());
    knncolle::VptreeBuilder<int, double, double> vb(eucdist);
    auto vptr = vb.build_known_unique(mat); // trying the overrides for some variety.
    knncolle::BruteforceBuilder<int, double, double> bb(eucdist);
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
    auto eucdist = std::make_shared<knncolle::EuclideanDistance<double, double> >();

    knncolle::VptreeBuilder<int, double, double> vb(eucdist);
    auto vptr = vb.build_known_shared(knncolle::SimpleMatrix<int, double>(ndim, nobs, data.data())); // trying the overrides for some variety.
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

            auto num = vsptr->search_all(x, new_threshold, &output_i, &output_d);
            EXPECT_EQ(output_i, ref_i); 
            EXPECT_EQ(output_d, ref_d);
            EXPECT_EQ(num, ref_i.size());

            vsptr->search_all(x, new_threshold, NULL, &output_d);
            EXPECT_EQ(output_d, ref_d);
            vsptr->search_all(x, new_threshold, &output_i, NULL);
            EXPECT_EQ(output_i, ref_i);

            auto num2 = vsptr->search_all(x, new_threshold, NULL, NULL);
            EXPECT_EQ(num, num2);
        }

        {
            auto ptr = data.data() + x * ndim;
            vsptr->search(ptr, k, &ref_i, &ref_d);
            double new_threshold = ref_d.back() * 0.99;
            while (ref_d.size() && ref_d.back() > new_threshold) {
                ref_d.pop_back();
                ref_i.pop_back();
            }

            auto num = vsptr->search_all(ptr, new_threshold, &output_i, &output_d);
            EXPECT_EQ(output_i, ref_i);
            EXPECT_EQ(output_d, ref_d);
            EXPECT_EQ(num, ref_i.size());

            vsptr->search_all(ptr, new_threshold, NULL, &output_d);
            EXPECT_EQ(output_d, ref_d);
            vsptr->search_all(ptr, new_threshold, &output_i, NULL);
            EXPECT_EQ(output_i, ref_i);

            auto num2 = vsptr->search_all(ptr, new_threshold, NULL, NULL);
            EXPECT_EQ(num, num2);
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

    auto eucdist = std::make_shared<knncolle::EuclideanDistance<double, double> >();
    knncolle::VptreeBuilder<int, double, double> bb(eucdist);
    int actual_nobs = nobs * duplication;
    auto bptr = bb.build_unique(knncolle::SimpleMatrix<int, double>(ndim, actual_nobs, dup.data()));
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

    auto eucdist = std::make_shared<knncolle::EuclideanDistance<double, double> >();
    knncolle::VptreeBuilder<int, double, double> vb(eucdist);
    auto vptr = vb.build_unique(knncolle::SimpleMatrix<int, double>(ndim, nobs, data.data()));
    auto vsptr = vptr->initialize();

    std::vector<int> res_i(10);
    std::vector<double> res_d(10);
    std::vector<double> target(ndim);
    vsptr->search(target.data(), 0, &res_i, &res_d);
    EXPECT_TRUE(res_i.empty());
    EXPECT_TRUE(res_d.empty());

    res_i.resize(10);
    res_d.resize(10);
    vsptr->search(target.data(), 10, &res_i, &res_d);
    EXPECT_TRUE(res_i.empty());
    EXPECT_TRUE(res_d.empty());

    res_i.resize(10);
    res_d.resize(10);
    EXPECT_EQ(vsptr->search_all(target.data(), 0, &res_i, &res_d), 0);
    EXPECT_TRUE(res_i.empty());
    EXPECT_TRUE(res_d.empty());

    // For coverage purposes:
    vsptr->search(target.data(), 0, NULL, NULL);
}

TEST(Vptree, Ties) {
    int ndim = 5;
    int nobs = 10;
    std::vector<double> data(ndim * nobs, 1);
    std::fill(data.begin() + nobs * ndim / 2, data.end(), 2);
    const double delta = std::sqrt(ndim);

    auto eucdist = std::make_shared<knncolle::EuclideanDistance<double, double> >();
    knncolle::VptreeBuilder<int, double, double> vb(eucdist);
    auto vptr = vb.build_unique(knncolle::SimpleMatrix<int, double>(ndim, nobs, data.data()));
    auto vsptr = vptr->initialize();
    std::vector<int> output_indices;
    std::vector<double> output_distances;

    // Check that ties are broken in a stable manner.
    {
        vsptr->search(0, 6, &output_indices, &output_distances);
        std::vector<int> expected_i { 1, 2, 3, 4, 5, 6 };
        EXPECT_EQ(output_indices, expected_i);
        std::vector<double> expected_d { 0, 0, 0, 0, delta, delta };
        EXPECT_EQ(output_distances, expected_d);
    }

    {
        vsptr->search(4, 5, &output_indices, &output_distances);
        std::vector<int> expected_i { 0, 1, 2, 3, 5 };
        EXPECT_EQ(output_indices, expected_i);
        std::vector<double> expected_d { 0, 0, 0, 0, delta };
        EXPECT_EQ(output_distances, expected_d);
    }

    {
        vsptr->search(5, 3, &output_indices, &output_distances);
        std::vector<int> expected_i { 6, 7, 8 };
        EXPECT_EQ(output_indices, expected_i);
        std::vector<double> expected_d { 0, 0, 0 };
        EXPECT_EQ(output_distances, expected_d);
    }

    {
        vsptr->search(9, 7, &output_indices, &output_distances);
        std::vector<int> expected_i { 5, 6, 7, 8, 0, 1, 2 };
        EXPECT_EQ(output_indices, expected_i);
        std::vector<double> expected_d { 0, 0, 0, 0, delta, delta, delta };
        EXPECT_EQ(output_distances, expected_d);
    }
}
