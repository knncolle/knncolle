#ifndef NEIGHBOR_QUEUE_HPP
#define NEIGHBOR_QUEUE_HPP

#include <queue>
#include "utils.hpp"

namespace knncolle {

/* The neighbor_queue class is a priority queue that contains indices and
 * distances in decreasing order from the top of the queue. Existing elements
 * are displaced by incoming elements that have shorter distances, thus making
 * it a useful data structure for retaining the k-nearest neighbors.
 *
 * We augment a normal priority queue with some extra features to:
 *
 * - remove self-matches for kNN searches. In such cases, the size of the queue
 *   is set to k+1 during the search, and any self-match is removed when the
 *   neighbors are reported to yield exactly k neighbors.  We cannot simply
 *   remove the closest neighbor in case of duplicates.
 * - warn about ties for exact searches. In such cases, the size of the queue
 *   is set to k+1 search. Reporting will check for tied distances among queue
 *   elements and emit one R warning per lifetime of the queue (to avoid
 *   saturating the warning counter in practical settings).
 *
 * Both of these options can be applied together, in which case the size of the
 * queue is set to k+2, any self-matches are removed, and distances are
 * searched for ties.
 */

class NeighborQueue {
public:
    NeighborQueue(NumNeighbors_t k, bool t) : ties(t), self(false) {
        base_setup(k);
        return;
    }

    NeighborQueue(CellIndex_t s, NumNeighbors_t k, bool t) : ties(t), self(true) , self_dex(s) {
        base_setup(k);
        return;
    }

    void add(CellIndex_t i, double d) {
        if (!full) {
            nearest.push(NeighborPoint(d, i));
            if (static_cast<NumNeighbors_t>(nearest.size())==check_k) {
                full=true;
            }
        } else if (d < limit()) {
            nearest.push(NeighborPoint(d, i));
            nearest.pop();
        }
        return;
    }

    bool is_full() const {
        return full;
    }

    double limit() const {
        return nearest.top().first;
    }

    bool report(std::vector<CellIndex_t>& indices, std::vector<double>& distances, bool report_indices, bool report_distances) {
        bool has_ties = false;
        indices.clear();
        distances.clear();
        if (nearest.empty()) {
            return has_ties;
        }

        // If 'self=false', then it never enters the !found_self clause below, which is the correct behaviour.
        bool found_self=!self; 

        while (!nearest.empty()) {
            if (!found_self && nearest.top().second==self_dex) {
                nearest.pop();
                found_self=true;
                continue;
            }
            if (report_indices) {
                indices.push_back(nearest.top().second);
            }
            if (report_distances || ties) {
                distances.push_back(nearest.top().first);
            }
            nearest.pop();
        }

        // We use push_back + reverse to give us sorting in increasing order;
        // this is nicer than push_front() for std::vectors.
        if (!indices.empty()) {
            std::reverse(indices.begin(), indices.end());
        }
        if (!distances.empty()) {
            std::reverse(distances.begin(), distances.end());
        }

        // Getting rid of the last entry to get the 'k' nearest neighbors, if 'self' was not in the queue.
        if (self && !found_self) {
            if (!indices.empty()) { 
                indices.pop_back();
            }
            if (!distances.empty()) {
                distances.pop_back();
            }
        }

        if (ties) {
            for (size_t d=1; d<distances.size(); ++d) {
                if (distances[d-1] >= distances[d]) {
                    has_ties = true;
                    break;
                }
            }
        
            // We assume that the NN search was conducted with an extra neighbor if ties=true upon entry.
            // This is necessary to allow the above code to check for whether there is a tie at the boundary of the set.
            // It is now time to remove this extra neighbor which should lie at the end of the set. The exception
            // is when we never actually fill up the queue, in which case we shouldn't do any popping.
            if (static_cast<NumNeighbors_t>(indices.size()) > n_neighbors) {
                indices.pop_back();
            }
            if (static_cast<NumNeighbors_t>(distances.size()) > n_neighbors) {
                distances.pop_back();
            }
        }

        return has_ties;
    } 
private:
    bool ties = true;
    bool self = false;
    CellIndex_t self_dex=0;
    NumNeighbors_t n_neighbors=0, check_k=1;
    bool full=false;

    void base_setup(NumNeighbors_t k) {
        n_neighbors=k;
        check_k=n_neighbors + self + ties;
        full=(check_k==0);
        return;
    }

    typedef std::pair<double, CellIndex_t> NeighborPoint;
    std::priority_queue<NeighborPoint> nearest;
};

}

#endif
