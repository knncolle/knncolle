#ifndef KNNCOLLE_NEIGHBOR_QUEUE_HPP
#define KNNCOLLE_NEIGHBOR_QUEUE_HPP

#include <queue>
#include <vector>
#include <algorithm>

/**
 * @file NeighborQueue.hpp
 * @brief Helper class to track nearest neighbors.
 */

namespace knncolle {

/**
 * @brief Helper class to track nearest neighbors.
 *
 * This is a priority queue that tracks the nearest neighbors of an observation of interest.
 * Specifically, it contains indices and distances of the `k` nearest neighbors, in decreasing order from the top of the queue.
 * When the queue is at capacity and new elements are added, existing elements will be displaced by incoming elements with shorter distances.
 *
 * This class is intended to be used in implementations of `Searcher::search()` to track the `k`-nearest neighbors.
 * When searching for neighbors of an existing observation in the dataset, it is recommended to search for the `k + 1` neighbors.
 * The appropriate `report()` overload can then be used to remove the observation of interest from its own neighbor list.
 *
 * @tparam Index_ Integer type for the observation indices.
 * @tparam Distance_ Floating point type for the distances.
 */
template<typename Index_, typename Distance_>
class NeighborQueue {
public:
    /**
     * Default constructor.
     * The maximum number of neighbors to be retained is 1; this can be changed by `reset()`.
     */
    NeighborQueue() = default;

public:    
    /**
     * Resets the queue to retain `k` neighbors.
     * Any existing neighbors in the queue are removed.
     *
     * @param k Maximum number of neighbors to retain.
     * This should be a positive integer.
     */
    void reset(Index_ k) {
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
    /**
     * @return Whether the queue is full.
     */
    bool is_full() const {
        return my_full;
    }

    /**
     * @return The distance of the `k`-th furthest neighbor, where `k` is the number of neighbors in `reset()`.
     * This should only be called if `is_full()` returns true.
     */
    Distance_ limit() const {
        return my_nearest.top().first;
   }

public:
    /**
     * Attempt to add a neighbor to the queue.
     * This will be a no-op if `is_full()` is true and `d > limit()`.
     *
     * @param i Index of the neighbor.
     * @param d Distance to the neighbor.
     */
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
    /**
     * Report the indices and distances of the nearest neighbors in the queue.
     * If the observation of interest is detected as its own neighbor, it will be removed from the output vectors.
     *
     * @param[out] output_indices Pointer to a vector in which to store the indices of the nearest neighbors, sorted by distance.
     * If `NULL`, the indices will not be reported.
     * @param[out] output_distances Pointer to a vector in which to store the (sorted) distances to the nearest neighbors.
     * If `NULL`, the distances will not be reported.
     * Otherwise, on output, this will have the same length as `*output_indices` and contain distances to each of those neighbors.
     * @param self Index of the observation of interest, i.e., for which neighbors are to be identified.
     * If present in the queue, this will be removed from `output_indices` and `output_distances`.
     */
    void report(std::vector<Index_>* output_indices, std::vector<Distance_>* output_distances, Index_ self) {
        // We expect that nearest is non-empty, as a search should at least
        // find 'self' (or duplicates thereof).
        auto num_expected = my_nearest.size() - 1u;
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
    /**
     * Report the indices and distances of the nearest neighbors in the queue.
     * It is assumed that the observation of interest is not detected as its own neighbor, presumably as it does not exist in the original input dataset.
     *
     * @param[out] output_indices Pointer to a vector in which to store the indices of the nearest neighbors, sorted by distance.
     * If `NULL`, the indices will not be reported.
     * @param[out] output_distances Pointer to a vector in which to store the (sorted) distances to the nearest neighbors.
     * If `NULL`, the distances will not be reported.
     * Otherwise, on output, this will have the same length as `*output_indices` and contain distances to each of those neighbors.
     */
    void report(std::vector<Index_>* output_indices, std::vector<Distance_>* output_distances) {
        auto position = my_nearest.size();
        if (output_indices) {
            output_indices->resize(position);
        }
        if (output_distances) {
            output_distances->resize(position);
        }

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
    bool my_full = false;
    std::priority_queue<std::pair<Distance_, Index_> > my_nearest;
    decltype(my_nearest.size()) my_neighbors = 1;
};

}

#endif
