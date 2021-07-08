#ifndef MATRIX_STORE_HPP
#define MATRIX_STORE_HPP

#include <vector>
#include <algorithm>

namespace knncolle {

template<bool COPY>
struct MatrixStore {
    MatrixStore(size_t n, const double* ptr) : store(COPY ? n : 0), reference(COPY ? store.data() : ptr) {
        if constexpr(COPY) {
            std::fill(ptr, ptr + n, store.begin());
        }
        return;
    }

    MatrixStore(const MatrixStore<COPY>& x) : store(x.store), reference(COPY ? store.data() : x.reference) {}

    MatrixStore& operator=(const MatrixStore<COPY>& x) {
        store = x.store;
        if constexpr(COPY) {
            reference = store.data();
        } else {
            reference = x.reference;
        }
        return *this;
    }

    MatrixStore(MatrixStore<COPY>&& x) : store(std::move(x.store)), reference(COPY ? store.data() : x.reference) {}

    MatrixStore& operator=(MatrixStore<COPY>&& x) {
        store = std::move(x.store);
        if constexpr(COPY) {
            reference = store.data();
        } else {
            reference = x.reference;
        }
        return *this;
    }

    ~MatrixStore() {}

    std::vector<double> store; 
    const double* reference;
};

}

#endif
