#include <gtest/gtest.h>
#include "knncolle/Vptree.hpp"
#include "knncolle/L2Normalized.hpp"

#include <vector>
#include <cmath>

#include "TestCore.hpp"

TEST(L2Normalized, Basic) {
    {
        std::vector<double> empty(5);
        std::vector<double> buffer(5, 123456);
        knncolle::internal::l2norm(empty.data(), 5, buffer.data());
        EXPECT_EQ(buffer, empty);
    }

    {
        std::vector<double> input { 3, 4 };
        std::vector<double> output(2);
        knncolle::internal::l2norm(input.data(), 2, output.data());
        EXPECT_FLOAT_EQ(output[0], 0.6);
        EXPECT_FLOAT_EQ(output[1], 0.8);
    }
}

class L2NormalizedMatrixTest : public TestCore, public ::testing::Test {
protected:
    void SetUp() {
        assemble({ 10, 20 });
    }
};

TEST_F(L2NormalizedMatrixTest, Matrix) {
    knncolle::SimpleMatrix<int, double> mat(ndim, nobs, data.data());
    knncolle::L2NormalizedMatrix<int, double, double> norm(mat);

    std::vector<double> mbuffer(ndim), nbuffer(ndim);
    auto mext = mat.new_extractor();
    auto next = norm.new_extractor();

    for (int i = 0; i < nobs; ++i) {
        auto mptr = mext->next();
        knncolle::internal::l2norm(mptr, ndim, mbuffer.data());
        auto nptr = next->next();
        std::copy_n(nptr, ndim, nbuffer.begin());
        EXPECT_EQ(mbuffer, nbuffer);
    }
}

class L2NormalizedTest : public TestCore, public ::testing::TestWithParam<std::tuple<std::tuple<int, int>, int> > {
protected:
    void SetUp() {
        assemble(std::get<0>(GetParam()));
    }

    static std::vector<double> l2normalize(int ndim, int nobs, std::vector<double> data) {
        for (int o = 0; o < nobs; ++o) {
            auto ptr = data.data() + o * static_cast<size_t>(ndim);
            double l2 = 0;
            for (int d = 0; d < ndim; ++d) {
                l2 += ptr[d] * ptr[d];
            }
            l2 = std::sqrt(l2);
            for (int d = 0; d < ndim; ++d) {
                ptr[d] /= l2;
            }
        }
        return data;
    }
};

TEST_P(L2NormalizedTest, Find) {
    int k = std::get<1>(GetParam());    
    auto normalized = l2normalize(ndim, nobs, data);
    auto eucdist = std::make_shared<knncolle::EuclideanDistance<double, double> >();

    auto bb = std::make_shared<knncolle::VptreeBuilder<int, double, double> >(eucdist);
    auto bptr = bb->build_unique(knncolle::SimpleMatrix<int, double>(ndim, nobs, normalized.data()));

    knncolle::L2NormalizedBuilder<int, double, double, double> lb(bb);
    auto lptr = lb.build_unique(knncolle::SimpleMatrix<int, double>(ndim, nobs, data.data()));
    EXPECT_EQ(ndim, lptr->num_dimensions());
    EXPECT_EQ(nobs, lptr->num_observations());

    auto bsptr = bptr->initialize();
    std::vector<int> output_i; 
    std::vector<double> output_d; 
    auto lsptr = lptr->initialize();
    std::vector<int> output2_i;
    std::vector<double> output2_d;

    for (int x = 0; x < nobs; ++x) {
        bsptr->search(x, k, &output_i, &output_d);
        lsptr->search(x, k, &output2_i, &output2_d);
        EXPECT_EQ(output_i, output2_i);
        EXPECT_EQ(output_d, output2_d);
    }
}

TEST_P(L2NormalizedTest, Query) {
    int k = std::get<1>(GetParam());    
    auto normalized = l2normalize(ndim, nobs, data);
    auto eucdist = std::make_shared<knncolle::EuclideanDistance<double, double> >();

    auto bb = std::make_shared<knncolle::VptreeBuilder<int, double, double> >(eucdist);
    auto bptr = bb->build_unique(knncolle::SimpleMatrix<int, double>(ndim, nobs, normalized.data()));

    knncolle::L2NormalizedBuilder<int, double, double, double> lb(bb);
    auto lptr = lb.build_shared(knncolle::SimpleMatrix<int, double>(ndim, nobs, data.data())); // shared pointer for some variety.
    EXPECT_EQ(ndim, lptr->num_dimensions());
    EXPECT_EQ(nobs, lptr->num_observations());

    auto bsptr = bptr->initialize();
    std::vector<int> output_i; 
    std::vector<double> output_d; 
    auto lsptr = lptr->initialize();
    std::vector<int> output2_i;
    std::vector<double> output2_d;

    for (int x = 0; x < nobs; ++x) {
        auto ptr = normalized.data() + x * ndim;
        bsptr->search(ptr, k, &output_i, &output_d);

        auto lptr = data.data() + x * ndim;
        lsptr->search(lptr, k, &output2_i, &output2_d);
        EXPECT_EQ(output_i, output2_i);
        EXPECT_EQ(output_d, output2_d);
    }
}

TEST_P(L2NormalizedTest, All) {
    int k = std::get<1>(GetParam());    
    auto normalized = l2normalize(ndim, nobs, data);
    auto eucdist = std::make_shared<knncolle::EuclideanDistance<double, double> >();

    auto bb = std::make_shared<knncolle::VptreeBuilder<int, double, double> >(eucdist);
    auto bptr = bb->build_unique(knncolle::SimpleMatrix<int, double>(ndim, nobs, normalized.data()));

    knncolle::L2NormalizedBuilder<int, double, double, double> lb(bb);
    auto lptr = lb.build_known_unique(knncolle::SimpleMatrix<int, double>(ndim, nobs, data.data())); // known override, for some variety.
    EXPECT_EQ(ndim, lptr->num_dimensions());
    EXPECT_EQ(nobs, lptr->num_observations());

    auto bsptr = bptr->initialize();
    std::vector<int> output_i; 
    std::vector<double> output_d; 
    auto lsptr = lptr->initialize();
    std::vector<int> output2_i;
    std::vector<double> output2_d;

    EXPECT_TRUE(lsptr->can_search_all());

    for (int x = 0; x < nobs; ++x) {
        bsptr->search(x, k, &output_i, &output_d);
        double new_threshold = output_d.back() * 0.99;

        auto count = bsptr->search_all(x, new_threshold, &output_i, &output_d);
        auto lcount = lsptr->search_all(x, new_threshold, &output2_i, &output2_d);
        EXPECT_EQ(count, lcount);
        EXPECT_EQ(output_i, output2_i);
        EXPECT_EQ(output_d, output2_d);

        auto ptr = normalized.data() + x * ndim;
        count = bsptr->search_all(ptr, new_threshold, &output_i, &output_d);
        auto lptr = data.data() + x * ndim;
        lcount = lsptr->search_all(lptr, new_threshold, &output2_i, &output2_d);
        EXPECT_EQ(count, lcount);
        EXPECT_EQ(output_i, output2_i);
        EXPECT_EQ(output_d, output2_d);
    }
}

INSTANTIATE_TEST_SUITE_P(
    L2Normalized,
    L2NormalizedTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::Values(10, 500), // number of observations
            ::testing::Values(5, 20) // number of dimensions
        ),
        ::testing::Values(3, 10, 20) // number of neighbors (one is greater than # observations, to test correct limiting)
    )
);

class L2NormalizedTypeTest : public TestCore, public ::testing::Test {
protected:
    void SetUp() {
        assemble({ 100, 10 });
    }
};

TEST_F(L2NormalizedTypeTest, NonBaseMatrix) {
    auto eucdist = std::make_shared<knncolle::EuclideanDistance<double, double> >();

    // Works correctly with a non-base Matrix_ class.
    typedef knncolle::L2NormalizedMatrix<int, double, double, knncolle::SimpleMatrix<int, double> > NormalizedMatrix;
    auto vb = std::make_shared<knncolle::VptreeBuilder<int, double, double, NormalizedMatrix> >(eucdist);
    knncolle::L2NormalizedBuilder<int, double, double, double, knncolle::SimpleMatrix<int, double> > lb(vb);

    auto lptr = lb.build_unique(knncolle::SimpleMatrix<int, double>(ndim, nobs, data.data()));
    EXPECT_EQ(lptr->num_observations(), nobs);
    EXPECT_EQ(lptr->num_dimensions(), ndim);

    // Comparing to the reference calculation.
    auto vbref  = std::make_shared<knncolle::VptreeBuilder<int, double, double> >(eucdist);
    knncolle::L2NormalizedBuilder<int, double, double, double> lbref(vbref);
    auto lrefptr = lbref.build_known_shared(knncolle::SimpleMatrix<int, double>(ndim, nobs, data.data())); // shared override, for some variety.

    auto lsptr = lptr->initialize();
    auto lsrefptr = lrefptr->initialize();
    std::vector<int> output_i, output2_i; 
    std::vector<double> output_d, output2_d; 

    int k = 5;
    for (int x = 0; x < nobs; ++x) {
        lsptr->search(x, k, &output_i, &output_d);
        lsrefptr->search(x, k, &output2_i, &output2_d);
        EXPECT_EQ(output_i, output2_i);
        EXPECT_EQ(output_d, output2_d);
    }
}

TEST_F(L2NormalizedTypeTest, NonIdenticalNormalized) {
    auto eucdist = std::make_shared<knncolle::EuclideanDistance<double, double> >();

    std::vector<double> rounded(data.begin(), data.end());
    for (auto& r : rounded) {
        r = std::round(r);
    }

    // Works correctly when Normalized_ != Data_.
    typedef knncolle::L2NormalizedMatrix<int, int, double> NormalizedMatrix;
    auto vb = std::make_shared<knncolle::VptreeBuilder<int, double, double, NormalizedMatrix> >(eucdist);
    knncolle::L2NormalizedBuilder<int, int, double, double> lb(vb);

    std::vector<int> idata(rounded.begin(), rounded.end());
    auto lptr = lb.build_unique(knncolle::SimpleMatrix<int, int>(ndim, nobs, idata.data()));
    EXPECT_EQ(lptr->num_observations(), nobs);
    EXPECT_EQ(lptr->num_dimensions(), ndim);

    // Comparing to the reference calculation.
    auto vbref  = std::make_shared<knncolle::VptreeBuilder<int, double, double> >(eucdist);
    knncolle::L2NormalizedBuilder<int, double, double, double> lbref(vbref);
    auto lrefptr = lbref.build_unique(knncolle::SimpleMatrix<int, double>(ndim, nobs, rounded.data()));

    auto lsptr = lptr->initialize();
    auto lsrefptr = lrefptr->initialize();
    std::vector<int> output_i, output2_i; 
    std::vector<double> output_d, output2_d; 

    int k = 5;
    for (int x = 0; x < nobs; ++x) {
        lsptr->search(x, k, &output_i, &output_d);
        lsrefptr->search(x, k, &output2_i, &output2_d);
        EXPECT_EQ(output_i, output2_i);
        EXPECT_EQ(output_d, output2_d);
    }
}
