#include <gtest/gtest.h>

#include "knncolle/Searcher.hpp"
#include "knncolle/distances.hpp"
#include "knncolle/Matrix.hpp"

#include <cstddef>
#include <memory>
#include <filesystem>

#include "TestCore.hpp"

class DistanceMetricSaveTest : public TestCore, public ::testing::Test {
protected:
    inline static std::filesystem::path savedir;

    static void SetUpTestSuite() {
        savedir = "save-distance-tests";
        std::filesystem::remove_all(savedir);
        std::filesystem::create_directory(savedir);
        assemble({ 2, 10 });

        knncolle::register_load_euclidean_distance<double, double>();
        knncolle::register_load_manhattan_distance<double, double>();
    }
};

TEST_F(DistanceMetricSaveTest, Euclidean) {
    knncolle::EuclideanDistance<double, double> metric;
    const auto dir = savedir / "euclidean";
    std::filesystem::create_directory(dir);
    metric.save(dir);

    std::shared_ptr<knncolle::DistanceMetric<double, double> > reloaded(knncolle::load_distance_metric_raw<double, double>(dir));

    const auto buf1 = data.data();
    const auto buf2 = data.data() + ndim;
    EXPECT_EQ(metric.raw(10, buf1, buf2), reloaded->raw(10, buf1, buf2));
}

TEST_F(DistanceMetricSaveTest, Manhattan) {
    knncolle::ManhattanDistance<double, double> metric;
    const auto dir = savedir / "manhattan";
    std::filesystem::create_directory(dir);
    metric.save(dir);

    std::shared_ptr<knncolle::DistanceMetric<double, double> > reloaded(knncolle::load_distance_metric_raw<double, double>(dir));

    const auto buf1 = data.data();
    const auto buf2 = data.data() + ndim;
    EXPECT_EQ(metric.raw(10, buf1, buf2), reloaded->raw(10, buf1, buf2));
}

class FakeDistanceMetric final : public knncolle::DistanceMetric<double, double> {
public:
    double raw(std::size_t, const double*, const double*) const { return 0; }
    double normalize(double) const { return 0; }
    double denormalize(double) const { return 0; }
};

TEST_F(DistanceMetricSaveTest, SaveError) {
    FakeDistanceMetric faker;
    EXPECT_EQ(faker.raw(0, NULL, NULL), 0);
    EXPECT_EQ(faker.normalize(0), 0);
    EXPECT_EQ(faker.denormalize(0), 0);
    EXPECT_ANY_THROW(faker.save("FOO"));
}

class FakeDistanceMetric2 final : public knncolle::DistanceMetric<double, double> {
public:
    double raw(std::size_t, const double*, const double*) const { return 0; }
    double normalize(double) const { return 0; }
    double denormalize(double) const { return 0; }
    void save(const std::filesystem::path& dir) const {
        knncolle::quick_save(dir / "DISTANCE", "foo", 3);
    }
};

TEST_F(DistanceMetricSaveTest, LoadError) {
    FakeDistanceMetric2 faker;
    EXPECT_EQ(faker.raw(0, NULL, NULL), 0);
    EXPECT_EQ(faker.normalize(0), 0);
    EXPECT_EQ(faker.denormalize(0), 0);

    const auto dir = savedir / "error";
    std::filesystem::create_directory(dir);
    faker.save(dir);

    std::string msg;
    try {
        knncolle::load_distance_metric_raw<double, double>(dir);
    } catch (knncolle::LoadDistanceMetricNotFoundError& e) {
        msg = e.what();
        EXPECT_EQ(e.get_distance(), "foo");
        EXPECT_EQ(e.get_path().filename().string(), "DISTANCE");
    }
    EXPECT_TRUE(msg.find("cannot find") != std::string::npos);
}

