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

    knncolle::KmknnBuilder<knncolle::ManhattanDistance> kb;
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

TEST_P(KmknnTest, AllEuclidean) {
    int k = std::get<1>(GetParam());    

    knncolle::KmknnBuilder<> kb;
    auto kptr = kb.build_unique(knncolle::SimpleMatrix(ndim, nobs, data.data()));
    auto ksptr = kptr->initialize();
    std::vector<int> output_i, ref_i;
    std::vector<double> output_d, ref_d;

    EXPECT_TRUE(ksptr->can_search_all());

    for (int x = 0; x < nobs; ++x) {
        {
            ksptr->search(x, k, &ref_i, &ref_d);
            double new_threshold = ref_d.back() * 0.99;
            while (ref_d.size() && ref_d.back() > new_threshold) {
                ref_d.pop_back();
                ref_i.pop_back();
            }

            auto num = ksptr->search_all(x, new_threshold, &output_i, &output_d);
            EXPECT_EQ(output_i, ref_i); 
            EXPECT_EQ(output_d, ref_d);
            EXPECT_EQ(num, ref_i.size());

            ksptr->search_all(x, new_threshold, NULL, &output_d);
            EXPECT_EQ(output_d, ref_d);
            ksptr->search_all(x, new_threshold, &output_i, NULL);
            EXPECT_EQ(output_i, ref_i);

            auto num2 = ksptr->search_all(x, new_threshold, NULL, NULL);
            EXPECT_EQ(num, num2);
        }

        {
            auto ptr = data.data() + x * ndim;
            ksptr->search(ptr, k, &ref_i, &ref_d);
            double new_threshold = ref_d.back() * 0.99;
            while (ref_d.size() && ref_d.back() > new_threshold) {
                ref_d.pop_back();
                ref_i.pop_back();
            }

            auto num = ksptr->search_all(ptr, new_threshold, &output_i, &output_d);
            EXPECT_EQ(output_i, ref_i);
            EXPECT_EQ(output_d, ref_d);
            EXPECT_EQ(num, ref_i.size());

            ksptr->search_all(ptr, new_threshold, NULL, &output_d);
            EXPECT_EQ(output_d, ref_d);
            ksptr->search_all(ptr, new_threshold, &output_i, NULL);
            EXPECT_EQ(output_i, ref_i);

            auto num2 = ksptr->search_all(ptr, new_threshold, NULL, NULL);
            EXPECT_EQ(num, num2);
        }
    }
}

TEST_P(KmknnTest, AllManhattan) {
    int k = std::get<1>(GetParam());    

    // Using Manhattan to test that denormalization is done correctly.
    knncolle::KmknnBuilder<knncolle::ManhattanDistance> kb;
    auto kptr = kb.build_unique(knncolle::SimpleMatrix(ndim, nobs, data.data()));
    auto ksptr = kptr->initialize();
    std::vector<int> output_i, ref_i;
    std::vector<double> output_d, ref_d;

    EXPECT_TRUE(ksptr->can_search_all());

    for (int x = 0; x < nobs; ++x) {
        ksptr->search(x, k, &ref_i, &ref_d);
        double new_threshold = ref_d.back() * 0.99;
        while (ref_d.size() && ref_d.back() > new_threshold) {
            ref_d.pop_back();
            ref_i.pop_back();
        }

        auto num = ksptr->search_all(x, new_threshold, &output_i, &output_d);
        EXPECT_EQ(output_i, ref_i); 
        EXPECT_EQ(output_d, ref_d);
        EXPECT_EQ(num, ref_i.size());
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
    // Duplicate tests also check that KMKNN handles zero-size clusters
    // correctly when these clusters occur after all other clusters. With the
    // default kmeans++ initialization, the trailing clusters will be empty if
    // 'k' is larger than the number of unique points.
    //
    // Note that we don't consider zero-size clusters that are intermingled
    // with non-zero-size clusters; see the SkipEmpty test below for that.

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

class KmknnMiscTest : public TestCore, public ::testing::Test {
protected:
    void SetUp() {
        assemble({ 100, 5 });
    }
};

TEST_F(KmknnMiscTest, Options) {
    knncolle::KmknnBuilder<> kb;
    EXPECT_FALSE(kb.get_options().initialize_algorithm);
    EXPECT_FALSE(kb.get_options().refine_algorithm);

    knncolle::KmknnOptions<> opt;
    opt.initialize_algorithm.reset(new kmeans::InitializeRandom<>);
    opt.refine_algorithm.reset(new kmeans::RefineLloyd<>);

    knncolle::KmknnBuilder<> kb2(opt); // test the constructor.
    EXPECT_TRUE(kb2.get_options().initialize_algorithm);
    EXPECT_TRUE(kb2.get_options().refine_algorithm);

    knncolle::SimpleMatrix<int, int, double> mat(ndim, nobs, data.data());
    auto kptr = kb.build_unique(mat);
    auto kptr2 = kb2.build_unique(mat);

    std::vector<int> kres_i, kres2_i;
    std::vector<double> kres_d, kres2_d;
    auto ksptr = kptr->initialize();
    auto ksptr2 = kptr2->initialize();

    for (int x = 0; x < nobs; ++x) {
        ksptr->search(x, 5, &kres_i, &kres_d);
        ksptr2->search(x, 5, &kres2_i, &kres2_d);
        EXPECT_EQ(kres_i, kres2_i);
        EXPECT_EQ(kres_d, kres2_d);
    }
}

template<class Matrix_ = kmeans::SimpleMatrix<double, int>, typename Cluster_ = int, typename Float_ = double>
struct InitializeNonsense : public kmeans::InitializeRandom<Matrix_, Cluster_, Float_> {
    Cluster_ run(const Matrix_& data, Cluster_ ncenters, Float_* centers) const {
        auto available = kmeans::InitializeRandom<Matrix_, Cluster_, Float_>::run(data, ncenters, centers);
        std::fill_n(centers, data.num_dimensions(), 100000000); // first one is nonsensically far away.
        return available;
    }
};

TEST_F(KmknnMiscTest, SkipEmptyClusters) {
    // We test the code that skips empty clusters in the constructor when these
    // clusters occur before a non-empty cluster. We do so by forcing the first
    // cluster to be empty by making its center ridiculous.

    knncolle::KmknnBuilder<> kb;
    auto& opt = kb.get_options();
    opt.initialize_algorithm.reset(new InitializeNonsense<>); // nothing will be assigned to the first cluster. 
    kmeans::RefineLloydOptions ll_opt;
    ll_opt.max_iterations = 1;
    opt.refine_algorithm.reset(new kmeans::RefineLloyd(ll_opt)); // no iterations so cluster centers can't be changed during refinement.

    knncolle::BruteforceBuilder<> bb;

    knncolle::SimpleMatrix<int, int, double> mat(ndim, nobs, data.data());
    auto kptr = kb.build_unique(mat);
    auto bptr = bb.build_unique(mat);

    std::vector<int> kres_i, ref_i;
    std::vector<double> kres_d, ref_d;
    auto bsptr = bptr->initialize();
    auto ksptr = kptr->initialize();

    for (int x = 0; x < nobs; ++x) {
        bsptr->search(x, 4, &ref_i, &ref_d);
        ksptr->search(x, 4, &kres_i, &kres_d);
        EXPECT_EQ(kres_i, ref_i);
        EXPECT_EQ(kres_d, ref_d);
    }
}

TEST(Kmknn, Empty) {
    int ndim = 5;
    int nobs = 0;
    std::vector<double> data;

    knncolle::KmknnBuilder<> kb;
    auto kptr = kb.build_unique(knncolle::SimpleMatrix(ndim, nobs, data.data()));
    auto ksptr = kptr->initialize();
    std::vector<int> res_i(10);
    std::vector<double> res_d(10);

    std::vector<double> target(ndim);
    ksptr->search(target.data(), 0, &res_i, &res_d);
    EXPECT_TRUE(res_i.empty());
    EXPECT_TRUE(res_d.empty());
}
