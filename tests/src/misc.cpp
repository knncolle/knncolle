#include <gtest/gtest.h>
#include "knncolle/Searcher.hpp"

#include <vector>
#include <random>

class DummyAlgorithm : public knncolle::Searcher<int, double> {
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
