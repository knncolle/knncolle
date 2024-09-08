#include <gtest/gtest.h>
#include "knncolle/Searcher.hpp"
#include "knncolle/distances.hpp"
#include "knncolle/NeighborQueue.hpp"

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

TEST(MockDistance, Basic) {
    std::vector<double> left(5);
    std::vector<double> right(5, 1);
    EXPECT_EQ(knncolle::MockDistance::template raw_distance<double>(left.data(), right.data(), 5), 5.0);
    EXPECT_EQ(knncolle::MockDistance::normalize(4.0), 4.0);
    EXPECT_EQ(knncolle::MockDistance::denormalize(4.0), 4.0);
}

TEST(NeighborQueue, Ties) {
    knncolle::internal::NeighborQueue<int, double> q;

    q.reset(5);
    for (int i = 10; i > 0; --i) {
        q.add(i, 1.0);
    }

    std::vector<int> output_indices;
    std::vector<double> output_distances;
    q.report(&output_indices, &output_distances);
    {
        std::vector<int> expected_i { 1, 2, 3, 4, 5 };
        EXPECT_EQ(output_indices, expected_i);
        EXPECT_EQ(output_distances, std::vector<double>(5, 1.0));
    }

    q.reset(6);
    for (int i = 11; i < 20; ++i) {
        q.add(i, 1.0);
    }

    q.report(&output_indices, &output_distances, 15);
    {
        std::vector<int> expected_i { 11, 12, 13, 14, 16 };
        EXPECT_EQ(output_indices, expected_i);
        EXPECT_EQ(output_distances, std::vector<double>(5, 1.0));
    }
}
