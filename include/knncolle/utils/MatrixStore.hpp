#ifndef MATRIX_STORE_HPP
#define MATRIX_STORE_HPP

#include <vector>
#include <algorithm>
#include <type_traits>

namespace knncolle {

template<typename DATA>
struct MatrixStore {
    template<typename INPUT>
    MatrixStore(const INPUT* ptr, size_t n, bool copy_ = false) : copy(copy_) {
        if (std::is_same<INPUT, DATA>::value && !copy) {
            reference = ptr;
        } else {
            copy = true;
            store.resize(n);
            std::copy(ptr, ptr + n, store.begin());
        }
        return;
    }

    MatrixStore& operator=(const MatrixStore<DATA>& x) {
        copy = x.copy;
        store = x.store;
        if (copy) {
            reference = store.data();
        } else {
            reference = x.reference;
        }
        return *this;
    }

    MatrixStore(MatrixStore<DATA>&& x) : copy(x.copy), store(std::move(x.store)), reference(copy ? store.data() : x.reference) {}

    MatrixStore& operator=(MatrixStore<DATA>&& x) {
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
    std::vector<DATA> store; 
    const DATA* reference;
};

}

#endif
