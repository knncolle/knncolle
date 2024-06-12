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
    static void sanity_checks(const std::vector<int>& indices, const std::vector<double>& distances) {
        EXPECT_TRUE(std::is_sorted(distances.begin(), distances.end())); // sorted by increasing distance.

        auto sorted = indices;
        std::sort(sorted.begin(), sorted.end());
        EXPECT_TRUE(std::adjacent_find(sorted.begin(), sorted.end()) == sorted.end()); // all neighbors are unique.
    }

    static void sanity_checks(const std::vector<int>& indices, const std::vector<double>& distances, int k, int self) { // for finding by index
        EXPECT_EQ(indices.size(), distances.size());
        EXPECT_EQ(indices.size(), std::min(k, nobs - 1));

        for (const auto& ix : indices) { // self is not in there.
            EXPECT_TRUE(ix != self);
        }

        sanity_checks(indices, distances);
    }
};

#endif
