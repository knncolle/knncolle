#include <gtest/gtest.h>

#include "knncolle/Searcher.hpp"
#include "knncolle/Matrix.hpp"
#include "knncolle/utils.hpp"

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

TEST(QuickSaveLoad, Errors) {
    {
        std::string errmsg;
        try {
            knncolle::quick_save("foo/bar", static_cast<double*>(NULL), 0);
        } catch (std::exception& e) {
            errmsg = e.what();
        }
        EXPECT_TRUE(errmsg.find("failed to open") != std::string::npos);
    }

    {
        std::string errmsg;
        try {
            knncolle::quick_load("foo/bar", static_cast<double*>(NULL), 0);
        } catch (std::exception& e) {
            errmsg = e.what();
        }
        EXPECT_TRUE(errmsg.find("failed to open") != std::string::npos);
    }

    {
        std::string errmsg;
        try {
            knncolle::quick_load_as_string("foo/bar");
        } catch (std::exception& e) {
            errmsg = e.what();
        }
        EXPECT_TRUE(errmsg.find("failed to open") != std::string::npos);
    }

    {
        auto path = std::filesystem::temp_directory_path() / "foobar-test";
        knncolle::quick_save(path.string(), "ABCD", 4);
        std::string errmsg;
        std::vector<char> buffer(10);
        EXPECT_ANY_THROW(knncolle::quick_load(path.string(), buffer.data(), buffer.size()));
    }
}
