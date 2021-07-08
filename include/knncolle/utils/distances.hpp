#ifndef KNNCOLLE_DISTANCES_HPP
#define KNNCOLLE_DISTANCES_HPP
#include <cmath>
#include "utils.hpp"

namespace knncolle {

namespace distances {

struct Euclidean {
    static double raw_distance(const double* x, const double* y, MatDim_t n) {
        double output = 0;
        for (MatDim_t i = 0; i < n; ++i, ++x, ++y) {
            output += ((*x) - (*y)) * ((*x) - (*y));
        }
        return output;
    }

    static double normalize(double raw) {
        return std::sqrt(raw);
    }
};

struct Manhattan {
    static double raw_distance(const double* x, const double* y, MatDim_t n) {
        double output = 0;
        for (MatDim_t i = 0; i < n; ++i, ++x, ++y) {
            output += std::abs(*x - *y);
        }
        return output;
    }

    static double normalize(double raw) {
        return raw;
    }
};

}

}

#endif
