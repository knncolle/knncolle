#ifndef TEST_CORE_HPP
#define TEST_CORE_HPP

#include <gtest/gtest.h>
#include <vector>
#include <random>
#include <tuple>

template<class PARAM>
class TestCore : public ::testing::TestWithParam<PARAM> {
protected:
    size_t nobs, ndim;
    std::vector<double> data;
protected:
    void assemble(const PARAM& param) {
        std::mt19937_64 rng(42);
        std::normal_distribution distr;

        nobs = std::get<0>(param);
        ndim = std::get<1>(param);
        data.resize(nobs * ndim);
        for (auto& d : data) {
            d = distr(rng);
        }

        return;
    }

    void sanity_checks(const std::vector<std::pair<int, double> >& results, int k) { // for finding by vector
        EXPECT_EQ(results.size(), std::min(k, (int)nobs));
        for (size_t i = 1; i < results.size(); ++i) { // check for sortedness.
            EXPECT_TRUE(results[i].second >= results[i-1].second);
        }
    }

    void sanity_checks(const std::vector<std::pair<int, double> >& results, int k, int index) const { // for finding by index
        EXPECT_EQ(results.size(), std::min(k, (int)nobs - 1));

        for (size_t i = 1; i < results.size(); ++i) { // check for sortedness.
            EXPECT_TRUE(results[i].second >= results[i-1].second);
        }

        for (const auto& res : results) { // self is not in there.
            EXPECT_TRUE(res.first != index);
        }
    }

    void compare_data(int index, const double* candidate) {
        for (size_t i = 0; i < ndim; ++i) {
            EXPECT_FLOAT_EQ(candidate[i], data[i + index * ndim]);
        }
    }
};

#endif
