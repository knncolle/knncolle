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
    std::vector<int> output_i, output_i0;
    std::vector<double> output_d, output_d0;
    auto bsptr2 = bptr2->initialize();
    std::vector<size_t> output2_i;
    std::vector<float> output2_d;

    for (int x = 0; x < nobs; ++x) {
        bsptr->search(x, k, &output_i, &output_d);
        sanity_checks(output_i, output_d, k, x);

        // Checking that it behaves with a NULL in either argument.
        bsptr->search(x, k, NULL, &output_d0);
        EXPECT_EQ(output_d, output_d0);
        bsptr->search(x, k, &output_i0, NULL);
        EXPECT_EQ(output_i, output_i0);

        bsptr2->search(x, k, &output2_i, &output2_d);
        EXPECT_EQ(output_i.size(), output2_i.size());
        EXPECT_EQ(output_d.size(), output2_d.size());
        for (size_t i = 0; i < output_i.size(); ++i) {
            EXPECT_EQ(output2_i[i], output_i[i]);
            EXPECT_FLOAT_EQ(output2_d[i], output_d[i]);
        }
    }
}

TEST_P(BruteforceTest, FindManhattan) {
    int k = std::get<1>(GetParam());    

    knncolle::BruteforceBuilder<knncolle::ManhattanDistance> bb;
    auto bptr = bb.build_unique(knncolle::SimpleMatrix(ndim, nobs, data.data()));

    auto bsptr = bptr->initialize();
    std::vector<int> output_i;
    std::vector<double> output_d;

    for (int x = 0; x < nobs; ++x) {
        bsptr->search(x, k, &output_i, &output_d);
        sanity_checks(output_i, output_d, k, x);
    }
}

TEST_P(BruteforceTest, QueryEuclidean) {
    int k = std::get<1>(GetParam());    

    knncolle::BruteforceBuilder<> bb;
    auto bptr = bb.build_shared(knncolle::SimpleMatrix(ndim, nobs, data.data())); // building a shared one for some variety.

    auto bsptr = bptr->initialize();
    std::vector<int> query_i, query_i0, ref_i;
    std::vector<double> query_d, query_d0, ref_d;

    for (int x = 0; x < nobs; ++x) {
        auto ptr = data.data() + x * ndim;
        bsptr->search(ptr, k + 1, &query_i, &query_d);
        EXPECT_EQ(query_i[0], x);
        EXPECT_EQ(query_d[0], 0);
        sanity_checks(query_i, query_d);

        // Same behavior with the NULLs.
        bsptr->search(ptr, k + 1, &query_i0, NULL);
        EXPECT_EQ(query_i, query_i0);
        bsptr->search(ptr, k + 1, NULL, &query_d0);
        EXPECT_EQ(query_d, query_d0);

        query_i.erase(query_i.begin());
        query_d.erase(query_d.begin());
        bsptr->search(x, k, &ref_i, &ref_d);
        EXPECT_EQ(ref_i, query_i);
        EXPECT_EQ(ref_d, query_d);
    }
}

TEST_P(BruteforceTest, AllEuclidean) {
    int k = std::get<1>(GetParam());    

    knncolle::BruteforceBuilder<> bb;
    auto bptr = bb.build_unique(knncolle::SimpleMatrix(ndim, nobs, data.data()));
    auto bsptr = bptr->initialize();
    std::vector<int> output_i, ref_i;
    std::vector<double> output_d, ref_d;

    EXPECT_TRUE(bsptr->can_search_all());

    for (int x = 0; x < nobs; ++x) {
        {
            bsptr->search(x, k, &ref_i, &ref_d);
            double new_threshold = ref_d.back() * 0.99;
            while (ref_d.size() && ref_d.back() > new_threshold) {
                ref_d.pop_back();
                ref_i.pop_back();
            }

            bsptr->search_all(x, new_threshold, &output_i, &output_d);
            EXPECT_EQ(output_i, ref_i); 
            EXPECT_EQ(output_d, ref_d);

            bsptr->search_all(x, new_threshold, NULL, &output_d);
            EXPECT_EQ(output_d, ref_d);
            bsptr->search_all(x, new_threshold, &output_i, NULL);
            EXPECT_EQ(output_i, ref_i);
        }

        {
            auto ptr = data.data() + x * ndim;
            bsptr->search(ptr, k, &ref_i, &ref_d);
            double new_threshold = ref_d.back() * 0.99;
            while (ref_d.size() && ref_d.back() > new_threshold) {
                ref_d.pop_back();
                ref_i.pop_back();
            }

            bsptr->search_all(ptr, new_threshold, &output_i, &output_d);
            EXPECT_EQ(output_i, ref_i);
            EXPECT_EQ(output_d, ref_d);

            bsptr->search_all(ptr, new_threshold, NULL, &output_d);
            EXPECT_EQ(output_d, ref_d);
            bsptr->search_all(ptr, new_threshold, &output_i, NULL);
            EXPECT_EQ(output_i, ref_i);
        }
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

class BruteforceDuplicateTest : public TestCore, public ::testing::TestWithParam<int> {
protected:
    void SetUp() {
        assemble({ 5, 3 });
    }
};

TEST_P(BruteforceDuplicateTest, Basic) {
    // Checking for correct elimination of self when reporting from a
    // NeighborQueue, while in the presence of many duplicates that could push
    // out 'self' from the results.

    int duplication = 10;
    std::vector<double> dup;
    for (int d = 0; d < duplication; ++d) {
        dup.insert(dup.end(), data.begin(), data.end());
    }

    knncolle::BruteforceBuilder<> bb;
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
    Bruteforce,
    BruteforceDuplicateTest,
    ::testing::Values(3, 10, 20) // number of neighbors
);
