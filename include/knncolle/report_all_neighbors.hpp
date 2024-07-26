#ifndef KNNCOLLE_REPORT_ALL_NEIGHBORS_HPP
#define KNNCOLLE_REPORT_ALL_NEIGHBORS_HPP

namespace knncolle {

namespace internal {

// These functions clean up the output for the search_all() functions.

template<bool do_indices_, bool do_distances_, typename Float_, typename Index_>
void report_all_neighbors_preamble(std::vector<std::pair<Float_, Index_> >& all_neighbors, std::vector<Index_>* output_indices, std::vector<Float_>* output_distances) {
    std::sort(all_neighbors.begin(), all_neighbors.end());
    if constexpr(do_indices_) {
        output_indices->clear();
        output_indices->reserve(all_neighbors.size() - 1);
    }
    if constexpr(do_distances_) {
        output_distances->clear();
        output_distances->reserve(all_neighbors.size() - 1);
    }
}

template<bool do_indices_, bool do_distances_, typename Float_, typename Index_>
void report_all_neighbors_raw(std::vector<std::pair<Float_, Index_> >& all_neighbors, std::vector<Index_>* output_indices, std::vector<Float_>* output_distances, Index_ i) {
    report_all_neighbors_preamble<do_indices_, do_distances_>(all_neighbors, output_indices, output_distances);
    for (const auto& an : all_neighbors) {
        if (an.second != i) {
            if constexpr(do_indices_) {
                output_indices->push_back(an.second);
            }
            if constexpr(do_distances_) {
                output_distances->push_back(an.first);
            }
        }
    }
}

template<typename Float_, typename Index_>
void report_all_neighbors(std::vector<std::pair<Float_, Index_> >& all_neighbors, std::vector<Index_>* output_indices, std::vector<Float_>* output_distances, Index_ i) {
    if (output_indices && output_distances) {
        report_all_neighbors_raw<true, true>(all_neighbors, output_indices, output_distances, i);
    } else if (output_indices) {
        report_all_neighbors_raw<true, false>(all_neighbors, output_indices, output_distances, i);
    } else if (output_distances) {
        report_all_neighbors_raw<false, true>(all_neighbors, output_indices, output_distances, i);
    }
}

template<bool do_indices_, bool do_distances_, typename Float_, typename Index_>
void report_all_neighbors_raw(std::vector<std::pair<Float_, Index_> >& all_neighbors, std::vector<Index_>* output_indices, std::vector<Float_>* output_distances) {
    report_all_neighbors_preamble<do_indices_, do_distances_>(all_neighbors, output_indices, output_distances);
    for (const auto& an : all_neighbors) {
        if constexpr(do_indices_) {
            output_indices->push_back(an.second);
        }
        if constexpr(do_distances_) {
            output_distances->push_back(an.first);
        }
    }
}

template<typename Float_, typename Index_>
void report_all_neighbors(std::vector<std::pair<Float_, Index_> >& all_neighbors, std::vector<Index_>* output_indices, std::vector<Float_>* output_distances) {
    if (output_indices && output_distances) {
        report_all_neighbors_raw<true, true>(all_neighbors, output_indices, output_distances);
    } else if (output_indices) {
        report_all_neighbors_raw<true, false>(all_neighbors, output_indices, output_distances);
    } else if (output_distances) {
        report_all_neighbors_raw<false, true>(all_neighbors, output_indices, output_distances);
    }
}

}

}

#endif
