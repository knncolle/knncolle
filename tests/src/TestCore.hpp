#ifndef TEST_CORE_HPP
#define TEST_CORE_HPP

#include <gtest/gtest.h>
#include <vector>
#include <random>
#include <tuple>

class TestCore {
protected:
    inline static int nobs, ndim;
    inline static std::vector<double> data;
    inline static std::tuple<int, int> last_params;

protected:
    static void assemble(const std::tuple<int, int>& param) {
        if (param == last_params) {
            return;
        }
        last_params = param;

        nobs = std::get<0>(param);
        ndim = std::get<1>(param);

        std::mt19937_64 rng(nobs * 10 + ndim);
        std::normal_distribution distr;

        data.resize(nobs * ndim);
        for (auto& d : data) {
            d = distr(rng);
        }
    }

    template<class It_, class Rng_>
    static void fill_random(It_ start, It_ end, Rng_& eng) {
        std::normal_distribution distr;
        while (start != end) {
            *start = distr(eng);
            ++start;
        }
    }

protected:
    static void sanity_checks(const std::vector<std::pair<int, double> >& results, int k) { // for finding by vector
        EXPECT_EQ(results.size(), std::min(k, nobs));
        for (size_t i = 1; i < results.size(); ++i) { // check for sortedness.
            EXPECT_TRUE(results[i].second >= results[i-1].second);
        }
    }

    static void sanity_checks(const std::vector<std::pair<int, double> >& results, int k, int index) { // for finding by index
        EXPECT_EQ(results.size(), std::min(k, nobs - 1));

        for (size_t i = 1; i < results.size(); ++i) { // check for sortedness.
            EXPECT_TRUE(results[i].second >= results[i-1].second);
        }

        for (const auto& res : results) { // self is not in there.
            EXPECT_TRUE(res.first != index);
        }
    }
};

#endif
