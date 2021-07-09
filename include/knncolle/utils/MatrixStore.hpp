#ifndef MATRIX_STORE_HPP
#define MATRIX_STORE_HPP

#include <vector>
#include <algorithm>

namespace knncolle {

template<typename DTYPE>
struct MatrixStore {
    MatrixStore(const DTYPE* ptr) : copy(false), reference(ptr) {}

    MatrixStore(std::vector<DTYPE> val) : copy(true), store(std::move(val)), reference(store.data()) {}

    MatrixStore(const MatrixStore<DTYPE>& x) : copy(x.copy), store(x.store), reference(copy ? store.data() : x.reference) {}

    MatrixStore& operator=(const MatrixStore<DTYPE>& x) {
        copy = x.copy;
        store = x.store;
        if (copy) {
            reference = store.data();
        } else {
            reference = x.reference;
        }
        return *this;
    }

    MatrixStore(MatrixStore<DTYPE>&& x) : copy(x.copy), store(std::move(x.store)), reference(copy ? store.data() : x.reference) {}

    MatrixStore& operator=(MatrixStore<DTYPE>&& x) {
        copy = x.copy;
        store = std::move(x.store);
        if (copy) {
            reference = store.data();
        } else {
            reference = x.reference;
        }
        return *this;
    }

    ~MatrixStore() {}

    bool copy;
    std::vector<double> store; 
    const double* reference;
};

}

#endif
