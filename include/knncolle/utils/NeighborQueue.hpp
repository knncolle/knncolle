#ifndef NEIGHBOR_QUEUE_HPP
#define NEIGHBOR_QUEUE_HPP

#include <queue>
#include <vector>
#include <algorithm>

namespace knncolle {

template<typename INDEX_t, typename DATA_t>
using neighbor_queue = std::priority_queue<std::pair<DATA_t, INDEX_t> >;

template<typename INDEX_t, typename DISTANCE_t, class QUEUE>
inline std::vector<std::pair<INDEX_t, DISTANCE_t> > harvest_queue(QUEUE& nearest, bool check_self = false, INDEX_t self_index = 0) {
    std::vector<std::pair<INDEX_t, DISTANCE_t> > output;

    // If 'check_self=false', then it never enters the !found_self clause below, which is the correct behaviour.
    bool found_self=!check_self;

    while (!nearest.empty()) {
        if (!found_self && nearest.top().second==self_index) {
            nearest.pop();
            found_self=true;
            continue;
        }
        output.push_back(std::pair<INDEX_t, DISTANCE_t>(nearest.top().second, nearest.top().first));
        nearest.pop();
    }

    // We use push_back + reverse to give us sorting in increasing order;
    // this is nicer than push_front() for std::vectors.
    std::reverse(output.begin(), output.end());

    // Getting rid of the last entry to get the 'k' nearest neighbors, if
    // 'self' was not in the queue; this assumes that we searched on the k+1
    // neighbors. Note that the absence of the self implies that we have at
    // least 'k + 2' total observations, otherwise we must have picked up the
    // self_index; thus, we have guaranteed to have gotten 'k+1' non-self hits,
    // so the pop is always safe.
    if (check_self && !found_self) {
        output.pop_back();
    }

    return output;
}

/* The NeighborQueue class is a priority queue that contains indices and
 * distances in decreasing order from the top of the queue. Existing elements
 * are displaced by incoming elements that have shorter distances, thus making
 * it a useful data structure for retaining the k-nearest neighbors.
 */
template<typename INDEX_t = int, typename DATA_t = double>
class NeighborQueue {
public:
    NeighborQueue(int k) : n_neighbors(k), full(n_neighbors == 0) {}

    NeighborQueue(int k, INDEX_t self) : n_neighbors(k + 1), full(false), check_self(true), self_index(self) {}

    void add(INDEX_t i, DATA_t d) {
        if (!full) {
            nearest.push(std::make_pair(d, i));
            if (static_cast<int>(nearest.size()) == n_neighbors) {
                full=true;
            }
        } else if (d < limit()) {
            nearest.push(std::make_pair(d, i));
            nearest.pop();
        }
        return;
    }

    bool is_full() const {
        return full;
    }

    DATA_t limit() const {
        return nearest.top().first;
   }

    template<typename DISTANCE_t>
    std::vector<std::pair<INDEX_t, DISTANCE_t> > report() {
        return harvest_queue<INDEX_t, DISTANCE_t>(nearest, check_self, self_index);
    } 
private:
    int n_neighbors;
    bool full = false;
    bool check_self = false;
    INDEX_t self_index = 0;
    neighbor_queue<INDEX_t, DATA_t> nearest;
};

}

#endif
