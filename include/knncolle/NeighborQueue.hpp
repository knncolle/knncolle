#ifndef KNNCOLLE_NEIGHBOR_QUEUE_HPP
#define KNNCOLLE_NEIGHBOR_QUEUE_HPP

#include <queue>
#include <vector>
#include <algorithm>

namespace knncolle {

namespace internal {

template<typename Index_, typename Distance_>
void flush_output(std::vector<Index_>* output_indices, std::vector<Distance_>* output_distances, size_t n) {
    if (output_indices) {
        output_indices->clear();
        output_indices->resize(n);
    }
    if (output_distances) {
        output_distances->clear();
        output_distances->resize(n);
    }
}

/* The NeighborQueue class is a priority queue that contains indices and
 * distances in decreasing order from the top of the queue. Existing elements
 * are displaced by incoming elements that have shorter distances, thus making
 * it a useful data structure for retaining the k-nearest neighbors.
 */
template<typename Index_, typename Distance_>
class NeighborQueue {
public:
    NeighborQueue() = default;

public:    
    void reset(Index_ k) { // We expect that k > 0.
        my_neighbors = k;
        my_full = false;

        // Popping any existing elements out, just in case. This shouldn't
        // usually be necessary if report() was called as the queue should
        // already be completely exhausted, but sometimes report() is a no-op
        // or there might have been an intervening exception, etc.
        while (!my_nearest.empty()) {
            my_nearest.pop();
        }
    }

public:
    bool is_full() const {
        return my_full;
    }

    Distance_ limit() const { // this should only be called if 'is_full()' returns true.
        return my_nearest.top().first;
   }

public:
    void add(Index_ i, Distance_ d) {
        if (!my_full) {
            my_nearest.emplace(d, i);
            if (my_nearest.size() == my_neighbors) {
                my_full = true;
            }
        } else {
            my_nearest.emplace(d, i);
            my_nearest.pop();
        }
        return;
    }

public:
    void report(std::vector<Index_>* output_indices, std::vector<Distance_>* output_distances, Index_ self) {
        // We expect that nearest is non-empty, as a search should at least
        // find 'self' (or duplicates thereof).
        size_t num_expected = my_nearest.size() - 1;
        if (output_indices) {
            output_indices->clear();
            output_indices->reserve(num_expected);
        }
        if (output_distances) {
            output_distances->clear();
            output_distances->reserve(num_expected);
        }

        bool found_self = false;
        while (!my_nearest.empty()) {
            const auto& top = my_nearest.top();
            if (!found_self && top.second == self) {
                found_self = true;
            } else {
                if (output_indices) {
                    output_indices->push_back(top.second);
                }
                if (output_distances) {
                    output_distances->push_back(top.first);
                }
            }
            my_nearest.pop();
        }

        // We use push_back + reverse to give us sorting in increasing order;
        // this is nicer than push_front() for std::vectors.
        if (output_indices) {
            std::reverse(output_indices->begin(), output_indices->end());
        }
        if (output_distances) {
            std::reverse(output_distances->begin(), output_distances->end());
        }

        // Removing the most distance element if we couldn't find ourselves,
        // e.g., because there are too many duplicates.
        if (!found_self) {
            if (output_indices) {
                output_indices->pop_back();
            }
            if (output_distances) {
                output_distances->pop_back();
            }
        }
    } 

public:
    void report(std::vector<Index_>* output_indices, std::vector<Distance_>* output_distances) {
        size_t position = my_nearest.size();
        flush_output(output_indices, output_distances, position);

        while (!my_nearest.empty()) {
            const auto& top = my_nearest.top();
            --position;
            if (output_indices) {
                (*output_indices)[position] = top.second;
            }
            if (output_distances) {
                (*output_distances)[position] = top.first;
            }
            my_nearest.pop();
        }
    } 

private:
    size_t my_neighbors = 1;
    bool my_full = false;
    std::priority_queue<std::pair<Distance_, Index_> > my_nearest;
};

}

}

#endif
