#ifndef KNNCOLLE_NEIGHBOR_QUEUE_HPP
#define KNNCOLLE_NEIGHBOR_QUEUE_HPP

#include <queue>
#include <vector>
#include <algorithm>

namespace knncolle {

namespace internal {

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
    void reset(Index_ k) {
        my_neighbors = k;
        my_full = (my_neighbors == 0);

        // Popping any existing elements out, just in case. This shouldn't
        // usually be necessary if report() was called as the queue should
        // already be copletely exhausted.
        while (!my_nearest.empty()) {
            my_nearest.pop();
        }

        // Avoid crashing if 'limit()' is called on an always-empty queue.
        if (my_full) {
            my_nearest.emplace(0, 0); 
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
        } else if (d < limit()) {
            my_nearest.emplace(d, i);
            my_nearest.pop();
        }
        return;
    }

    void report(std::vector<std::pair<Index_, Distance_> >& output, Index_ self) {
        output.clear();
        output.reserve(my_nearest.size() - 1);

        bool found_self = false;
        while (!my_nearest.empty()) {
            const auto& top = my_nearest.top();
            if (!found_self && top.second == self) {
                found_self = true;
            } else {
                output.emplace_back(top.second, top.first);
            }
            my_nearest.pop();
        }

        // We use push_back + reverse to give us sorting in increasing order;
        // this is nicer than push_front() for std::vectors.
        std::reverse(output.begin(), output.end());

        // Removing the most distance element if we couldn't find ourselves,
        // e.g., because there are too many duplicates.
        if (!found_self) {
            output.pop_back();
        }
    } 

    void report(std::vector<std::pair<Index_, Distance_> >& output) {
        output.clear();
        output.reserve(my_nearest.size());

        while (!my_nearest.empty()) {
            const auto& top = my_nearest.top();
            output.emplace_back(top.second, top.first);
            my_nearest.pop();
        }

        // We use push_back + reverse to give us sorting in increasing order;
        // this is nicer than push_front() for std::vectors.
        std::reverse(output.begin(), output.end());
    } 

private:
    size_t my_neighbors = 0;
    bool my_full = true;
    std::priority_queue<std::pair<Distance_, Index_> > my_nearest;
};

}

}

#endif
