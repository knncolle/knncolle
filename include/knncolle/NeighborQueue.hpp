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
    NeighborQueue(Index_ k) : my_neighbors(k), my_full(my_neighbors == 0) {
        if (my_full) {
            nearest.emplace(0, 0); // avoid crashing if 'limit()' is called.
        }
    }

    bool is_full() const {
        return my_full;
    }

    Data_ limit() const { // this should only be called if 'is_full()' returns true.
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
            nearest.emplace(d, i);
            nearest.pop();
        }
        return;
    }

    std::vector<std::pair<Index_, Distance_> > report() {
        std::vector<std::pair<INDEX_t, DISTANCE_t> > output;

        while (!my_nearest.empty()) {
            const auto& top = my_nearest.top();
            output.push_back(std::pair<INDEX_t, DISTANCE_t>(top.second, top.first));
            my_nearest.pop();
        }

        // We use push_back + reverse to give us sorting in increasing order;
        // this is nicer than push_front() for std::vectors.
        std::reverse(output.begin(), output.end());
        return output;
    } 

private:
    size_t my_neighbors;
    bool my_full = false;
    std::priority_queue<std::pair<Data_, Index_> > my_nearest;
};

}

}

#endif