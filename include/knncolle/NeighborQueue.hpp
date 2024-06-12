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
        // already be completely exhausted, but sometimes report() is a no-op
        // or there might have been an intervening exception, etc.
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

private:
    template<bool do_indices_, bool do_distances_>
    void report_raw(std::vector<Index_>* output_indices, std::vector<Distance_>* output_distances, Index_ self) {
        if constexpr(do_indices_) {
            output_indices->clear();
            output_indices->reserve(my_nearest.size() - 1);
        }
        if constexpr(do_distances_) {
            output_distances->clear();
            output_distances->reserve(my_nearest.size() - 1);
        }

        bool found_self = false;
        while (!my_nearest.empty()) {
            const auto& top = my_nearest.top();
            if (!found_self && top.second == self) {
                found_self = true;
            } else {
                if constexpr(do_indices_) {
                    output_indices->push_back(top.second);
                }
                if constexpr(do_distances_) {
                    output_distances->push_back(top.first);
                }
            }
            my_nearest.pop();
        }

        // We use push_back + reverse to give us sorting in increasing order;
        // this is nicer than push_front() for std::vectors.
        if constexpr(do_indices_) {
            std::reverse(output_indices->begin(), output_indices->end());
        }
        if constexpr(do_distances_) {
            std::reverse(output_distances->begin(), output_distances->end());
        }

        // Removing the most distance element if we couldn't find ourselves,
        // e.g., because there are too many duplicates.
        if (!found_self) {
            if constexpr(do_indices_) {
                output_indices->pop_back();
            }
            if constexpr(do_distances_) {
                output_distances->pop_back();
            }
        }
    } 

public:
    void report(std::vector<Index_>* output_indices, std::vector<Distance_>* output_distances, Index_ self) {
        if (output_indices && output_distances) {
            report_raw<true, true>(output_indices, output_distances, self);
        } else if (output_indices) {
            report_raw<true, false>(output_indices, output_distances, self);
        } else if (output_distances) {
            report_raw<false, true>(output_indices, output_distances, self);
        }
    }

private:
    template<bool do_indices_, bool do_distances_>
    void report_raw(std::vector<Index_>* output_indices, std::vector<Distance_>* output_distances) {
        size_t position = my_nearest.size();

        if constexpr(do_indices_) {
            output_indices->clear();
            output_indices->resize(position);
        }
        if constexpr(do_distances_) {
            output_distances->clear();
            output_distances->resize(position);
        }

        while (!my_nearest.empty()) {
            const auto& top = my_nearest.top();
            --position;
            if constexpr(do_indices_) {
                (*output_indices)[position] = top.second;
            }
            if constexpr(do_distances_) {
                (*output_distances)[position] = top.first;
            }
            my_nearest.pop();
        }
    } 

public:
    void report(std::vector<Index_>* output_indices, std::vector<Distance_>* output_distances) {
        if (output_indices && output_distances) {
            report_raw<true, true>(output_indices, output_distances);
        } else if (output_indices) {
            report_raw<true, false>(output_indices, output_distances);
        } else if (output_distances) {
            report_raw<false, true>(output_indices, output_distances);
        }
    }

private:
    size_t my_neighbors = 0;
    bool my_full = true;
    std::priority_queue<std::pair<Distance_, Index_> > my_nearest;
};

}

}

#endif
