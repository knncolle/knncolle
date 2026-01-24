#include <gtest/gtest.h>
#include "knncolle/load_prebuilt.hpp"

#include <filesystem>

#include "TestCore.hpp"

class LoadPrebuiltTest : public TestCore, public ::testing::Test {
protected:
    std::filesystem::path savedir;

    void SetUp() {
        savedir = "save-prebuilt-tests";
        std::filesystem::remove_all(savedir);
        std::filesystem::create_directory(savedir);

        assemble({ 50, 5 });
    }
};

TEST_F(LoadPrebuiltTest, BruteforceEuclidean) {
    auto eucdist = std::make_shared<knncolle::EuclideanDistance<double, double> >();
    knncolle::BruteforceBuilder<int, double, double> bb(eucdist);
    auto bptr = bb.build_unique(knncolle::SimpleMatrix<int, double>(ndim, nobs, data.data()));

    const auto prefix = (savedir / "bruteforce_euclidean_").string();
    bptr->save(prefix);

    auto reloaded = knncolle::load_prebuilt_shared<int, double, double>(prefix);
    std::vector<int> output_i, output_i2;
    std::vector<double> output_d, output_d2;

    auto searcher = bptr->initialize();
    auto researcher = reloaded->initialize();
    for (int x = 0; x < nobs; ++x) {
        searcher->search(x, 5, &output_i, &output_d);
        researcher->search(x, 5, &output_i2, &output_d2);
        EXPECT_EQ(output_i, output_i2);
        EXPECT_EQ(output_d, output_d2);
    }
}

TEST_F(LoadPrebuiltTest, BruteforceManhattan) {
    auto mandist = std::make_shared<knncolle::ManhattanDistance<double, float> >();
    knncolle::BruteforceBuilder<std::size_t, double, float> bb(mandist);
    auto bptr = bb.build_unique(knncolle::SimpleMatrix<std::size_t, double>(ndim, nobs, data.data()));

    const auto prefix = (savedir / "bruteforce_manhattan_").string();
    bptr->save(prefix);

    auto reloaded = knncolle::load_prebuilt_unique<std::size_t, double, float>(prefix);
    std::vector<std::size_t> output_i, output_i2;
    std::vector<float> output_d, output_d2;

    auto searcher = bptr->initialize();
    auto researcher = reloaded->initialize();
    for (int x = 0; x < nobs; ++x) {
        searcher->search(x, 10, &output_i, &output_d);
        researcher->search(x, 10, &output_i2, &output_d2);
        EXPECT_EQ(output_i, output_i2);
        EXPECT_EQ(output_d, output_d2);
    }
}

TEST_F(LoadPrebuiltTest, VptreeEuclidean) {
    auto eucdist = std::make_shared<knncolle::EuclideanDistance<double, double> >();
    knncolle::VptreeBuilder<int, double, double> vb(eucdist);
    auto vptr = vb.build_unique(knncolle::SimpleMatrix<int, double>(ndim, nobs, data.data()));

    const auto prefix = (savedir / "vptree_euclidean_").string();
    vptr->save(prefix);

    auto reloaded = knncolle::load_prebuilt_shared<int, double, double>(prefix);
    std::vector<int> output_i, output_i2;
    std::vector<double> output_d, output_d2;

    auto searcher = vptr->initialize();
    auto researcher = reloaded->initialize();
    for (int x = 0; x < nobs; ++x) {
        searcher->search(x, 5, &output_i, &output_d);
        researcher->search(x, 5, &output_i2, &output_d2);
        EXPECT_EQ(output_i, output_i2);
        EXPECT_EQ(output_d, output_d2);
    }
}

TEST_F(LoadPrebuiltTest, VptreeManhattan) {
    auto mandist = std::make_shared<knncolle::ManhattanDistance<double, float> >();
    knncolle::VptreeBuilder<std::size_t, double, float> vb(mandist);
    auto vptr = vb.build_unique(knncolle::SimpleMatrix<std::size_t, double>(ndim, nobs, data.data()));

    const auto prefix = (savedir / "vptree_manhattan_").string();
    vptr->save(prefix);

    auto reloaded = knncolle::load_prebuilt_unique<std::size_t, double, float>(prefix);
    std::vector<std::size_t> output_i, output_i2;
    std::vector<float> output_d, output_d2;

    auto searcher = vptr->initialize();
    auto researcher = reloaded->initialize();
    for (int x = 0; x < nobs; ++x) {
        searcher->search(x, 10, &output_i, &output_d);
        researcher->search(x, 10, &output_i2, &output_d2);
        EXPECT_EQ(output_i, output_i2);
        EXPECT_EQ(output_d, output_d2);
    }
}

class FakePrebuilt final : public knncolle::Prebuilt<int, double, double> {
public:
    int num_observations() const { return 0; }
    std::size_t num_dimensions() const { return 0; }
    std::unique_ptr<knncolle::Searcher<int, double, double> > initialize() const { return NULL; }
};

TEST_F(LoadPrebuiltTest, Errors) {
    const auto prefix = savedir / "error_";
    {
        auto dispatch = prefix;
        dispatch += "ALGORITHM";
        std::ofstream out(dispatch);
        out << "superfoobar";
    }

    std::string msg;
    try {
        knncolle::load_prebuilt_shared<int, double, double>(prefix.string());
    } catch (std::exception& e) {
        msg = e.what();
    }
    EXPECT_TRUE(msg.find("superfoobar") != std::string::npos);

    FakePrebuilt faker;
    EXPECT_EQ(faker.num_observations(), 0);
    EXPECT_EQ(faker.num_dimensions(), 0);
    EXPECT_FALSE(faker.initialize());
    EXPECT_ANY_THROW(faker.save("FOO"));
}
