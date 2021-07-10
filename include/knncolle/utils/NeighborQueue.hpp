#ifndef NEIGHBOR_QUEUE_HPP
#define NEIGHBOR_QUEUE_HPP

#include <queue>
#include <vector>

namespace knncolle {

template<typename ITYPE, typename DTYPE>
using neighbor_queue = std::priority_queue<std::pair<DTYPE, ITYPE> >;

template<class QUEUE, typename ITYPE, typename DTYPE>
inline void harvest_queue(QUEUE& nearest, std::vector<ITYPE>* indices, std::vector<DTYPE>* distances, bool check_self = false, ITYPE self_index = 0) {
    if (indices) {
        indices->clear();
    }
    if (distances) {
        distances->clear();
    }

    // If 'check_self=false', then it never enters the !found_self clause below, which is the correct behaviour.
    bool found_self=!check_self;

    while (!nearest.empty()) {
        if (!found_self && nearest.top().second==self_index) {
            nearest.pop();
            found_self=true;
            continue;
        }
        if (indices) {
            indices->push_back(nearest.top().second);
        }
        if (distances) {
            distances->push_back(nearest.top().first);
        }
        nearest.pop();
    }

    // We use push_back + reverse to give us sorting in increasing order;
    // this is nicer than push_front() for std::vectors.
    if (indices) {
        std::reverse(indices->begin(), indices->end());
    }
    if (distances) {
        std::reverse(distances->begin(), distances->end());
    }

    // Getting rid of the last entry to get the 'k' nearest neighbors, if 'self' was not in the queue.
    if (check_self && !found_self) {
        if (indices) {
            indices->pop_back();
        }
        if (distances) {
            distances->pop_back();
        }
    }

    return;
}

/* The NeighborQueue class is a priority queue that contains indices and
 * distances in decreasing order from the top of the queue. Existing elements
 * are displaced by incoming elements that have shorter distances, thus making
 * it a useful data structure for retaining the k-nearest neighbors.
 */
template<typename ITYPE = int, typename DTYPE = double>
class NeighborQueue {
public:
    NeighborQueue(int k) : n_neighbors(k), full(n_neighbors == 0) {}

    void add(ITYPE i, DTYPE d) {
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

    DTYPE limit() const {
        return nearest.top().first;
   }

    void report(std::vector<ITYPE>* indices, std::vector<DTYPE>* distances, bool check_self = false, ITYPE self_index = 0) {
        harvest_queue(nearest, indices, distances, check_self, self_index);
        return;
    } 
private:
    int n_neighbors;
    bool full = false;
    neighbor_queue<ITYPE, DTYPE> nearest;
};

}

#endif
