#include <gtest/gtest.h>
#include "knncolle/load_prebuilt.hpp"

#include <filesystem>
#include <memory>
#include <vector>
#include <cstddef>
#include <string>
#include <stdexcept>

#include "TestCore.hpp"

class LoadPrebuiltTest : public TestCore, public ::testing::Test {
protected:
    inline static std::filesystem::path savedir;

    static void SetUpTestSuite() {
        savedir = "save-prebuilt-tests";
        std::filesystem::remove_all(savedir);
        std::filesystem::create_directory(savedir);
        assemble({ 50, 5 });

        knncolle::register_load_bruteforce_prebuilt<int, double, double>();
        knncolle::register_load_vptree_prebuilt<int, double, double>();
        knncolle::register_load_euclidean_distance<double, double>();

        knncolle::register_load_bruteforce_prebuilt<std::size_t, double, float>();
        knncolle::register_load_vptree_prebuilt<std::size_t, double, float>();
        knncolle::register_load_manhattan_distance<double, float>();
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
    // Trying other types for some variety.
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
    // Trying other types for some variety.
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

TEST_F(LoadPrebuiltTest, L2NormalizedEuclidean) {
    auto& reg = knncolle::load_prebuilt_registry<int, double, double>(); 
    reg[knncolle::l2normalized_prebuilt_save_name] = [](const std::string& prefix) -> knncolle::Prebuilt<int, double, double>* {
        auto config = knncolle::load_l2normalized_prebuilt_types(prefix);
        EXPECT_EQ(config.normalized, knncolle::NumericType::DOUBLE);
        return knncolle::load_l2normalized_prebuilt<int, double, double, double>(prefix);
    };

    auto eucdist = std::make_shared<knncolle::EuclideanDistance<double, double> >();
    knncolle::L2NormalizedBuilder<int, double, double, double> l2b(std::make_shared<knncolle::VptreeBuilder<int, double, double> >(eucdist));
    auto l2ptr = l2b.build_unique(knncolle::SimpleMatrix<int, double>(ndim, nobs, data.data()));

    const auto prefix = (savedir / "vptree_l2norm_").string();
    l2ptr->save(prefix);

    auto reloaded = knncolle::load_prebuilt_unique<int, double, double>(prefix);
    std::vector<int> output_i, output_i2;
    std::vector<double> output_d, output_d2;

    auto searcher = l2ptr->initialize();
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
    } catch (knncolle::LoadPrebuiltNotFoundError& e) {
        msg = e.what();
        EXPECT_EQ(e.get_algorithm(), "superfoobar");
        EXPECT_FALSE(e.get_path().find("error_ALGORITHM") == std::string::npos);
    }
    EXPECT_TRUE(msg.find("superfoobar") != std::string::npos);

    FakePrebuilt faker;
    EXPECT_EQ(faker.num_observations(), 0);
    EXPECT_EQ(faker.num_dimensions(), 0);
    EXPECT_FALSE(faker.initialize());
    EXPECT_ANY_THROW(faker.save("FOO"));
}
