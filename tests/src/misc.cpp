#include <gtest/gtest.h>

#include "knncolle/Searcher.hpp"
#include "knncolle/distances.hpp"
#include "knncolle/Matrix.hpp"

#include <vector>
#include <random>
#include <filesystem>

#include "TestCore.hpp"

class DummyAlgorithm : public knncolle::Searcher<int, double, double> {
public:
    void search(int, int, std::vector<int>*, std::vector<double>*) {}
    void search(const double*, int, std::vector<int>*, std::vector<double>*) {}
};

TEST(SearcherDefaults, Basic) {
    DummyAlgorithm tmp;

    std::vector<int> tmp_i;
    std::vector<double> tmp_d;
    tmp.search(0, 0, &tmp_i, &tmp_d);
    tmp.search(static_cast<double*>(NULL), 0, &tmp_i, &tmp_d);

    EXPECT_FALSE(tmp.can_search_all());
    EXPECT_ANY_THROW(tmp.search_all(0, 1, &tmp_i, &tmp_d));
    EXPECT_ANY_THROW(tmp.search_all(static_cast<double*>(NULL), 1, &tmp_i, &tmp_d));
}

class SimpleMatrixTest : public TestCore, public ::testing::Test {
protected:
    void SetUp() {
        assemble({ 10, 20 });
    }
};

TEST_F(SimpleMatrixTest, Basic) {
    knncolle::SimpleMatrix<int, double> mat(ndim, nobs, data.data());
    std::vector<double> ref(ndim);
    std::vector<double> nbuffer(ndim);
    auto mext = mat.new_extractor();

    for (int i = 0; i < nobs; ++i) {
        std::copy_n(data.data() + static_cast<std::size_t>(i) * static_cast<std::size_t>(ndim), ndim, ref.data());
        std::copy_n(mext->next(), ndim, nbuffer.begin());
        EXPECT_EQ(ref, nbuffer);
    }
}

class DistanceMetricSaveTest : public TestCore, public ::testing::Test {
protected:
    std::filesystem::path savedir;
    const double* buf1, *buf2;

    void SetUp() {
        savedir = "save-distance-tests";
        std::filesystem::remove_all(savedir);
        std::filesystem::create_directory(savedir);

        assemble({ 2, 10 });
        buf1 = data.data();
        buf2 = data.data() + ndim;
    }
};

TEST_F(DistanceMetricSaveTest, Euclidean) {
    knncolle::EuclideanDistance<double, double> metric;
    const auto prefix = (savedir / "euclidean_").string();
    metric.save(prefix);
    std::shared_ptr<knncolle::DistanceMetric<double, double> > reloaded(knncolle::load_distance_metric_raw<double, double>(prefix));
    EXPECT_EQ(metric.raw(10, buf1, buf2), reloaded->raw(10, buf1, buf2));
}

TEST_F(DistanceMetricSaveTest, Manhattan) {
    knncolle::ManhattanDistance<double, double> metric;
    const auto prefix = (savedir / "manhattan_").string();
    metric.save(prefix);
    std::shared_ptr<knncolle::DistanceMetric<double, double> > reloaded(knncolle::load_distance_metric_raw<double, double>(prefix));
    EXPECT_EQ(metric.raw(10, buf1, buf2), reloaded->raw(10, buf1, buf2));
}

class FakeDistanceMetric final : public knncolle::DistanceMetric<double, double> {
public:
    double raw(std::size_t, const double*, const double*) const { return 0; }
    double normalize(double) const { return 0; }
    double denormalize(double) const { return 0; }
};

TEST_F(DistanceMetricSaveTest, Error) {
    FakeDistanceMetric faker;
    EXPECT_EQ(faker.raw(0, NULL, NULL), 0);
    EXPECT_EQ(faker.normalize(0), 0);
    EXPECT_EQ(faker.denormalize(0), 0);
    EXPECT_ANY_THROW(faker.save("FOO"));
}
