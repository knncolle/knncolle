#ifndef KNNCOLLE_L2_NORMALIZED_HP
#define KNNCOLLE_L2_NORMALIZED_HP

#include <vector>
#include <cmath>
#include <memory>

#include "Searcher.hpp"
#include "Prebuilt.hpp"
#include "Builder.hpp"
#include "MockMatrix.hpp"

/**
 * @file L2Normalized.hpp
 * @brief Wrapper for L2 normalization prior to search.
 */

namespace knncolle {

/**
 * @cond
 */
namespace internal {

template<typename Float_>
const Float_* l2norm(const Float_* ptr, size_t ndim, Float_* buffer) {
    Float_ l2 = 0;
    for (size_t d = 0; d < ndim; ++d) {
        auto val = ptr[d];
        l2 += val * val;
    }

    if (l2 == 0) {
        return ptr;
    }

    l2 = std::sqrt(l2);
    for (size_t d = 0; d < ndim; ++d) {
        buffer[d] = ptr[d] / l2;
    }
    return buffer;
}

}
/**
 * @endcond
 */

/**
 * @brief Wrapper around a search interface with L2 normalization.
 * 
 * This applies L2 normalization to each query vector before running `search()` and `search_all()`, typically for calculation of cosine distances.
 * Instances of this class are typically constructed with `L2NormalizedPrebuilt::initialize()`.
 *
 * @tparam Index_ Integer type for the indices.
 * For the output of `Builder::build`, this is set to `MockMatrix::index_type`.
 * @tparam Float_ Floating point type for the query data and output distances.
 */
template<typename Index_, typename Float_>
class L2NormalizedSearcher : public Searcher<Index_, Float_> {
public:
    /**
     * @param searcher Pointer to a `Searcher` class for the neighbor search that is to be wrapped.
     * @param num_dimensions Number of dimensions of the data.
     */
    L2NormalizedSearcher(std::unique_ptr<Searcher<Index_, Float_> > searcher, size_t num_dimensions) : 
        my_searcher(std::move(searcher)),
        buffer(num_dimensions)
    {}

private:
    std::unique_ptr<Searcher<Index_, Float_> > my_searcher;
    std::vector<Float_> buffer;

    /**
     * @cond
     */
public:
    void search(Index_ i, Index_ k, std::vector<Index_>* output_indices, std::vector<Float_>* output_distances) {
        my_searcher->search(i, k, output_indices, output_distances);
    }

    void search(const Float_* ptr, Index_ k, std::vector<Index_>* output_indices, std::vector<Float_>* output_distances) {
        auto normalized = internal::l2norm(ptr, buffer.size(), buffer.data());
        my_searcher->search(normalized, k, output_indices, output_distances);
    }

public:
    bool can_search_all() const {
        return my_searcher->can_search_all();
    }

    Index_ search_all(Index_ i, Float_ threshold, std::vector<Index_>* output_indices, std::vector<Float_>* output_distances) {
        return my_searcher->search_all(i, threshold, output_indices, output_distances);
    }

    Index_ search_all(const Float_* ptr, Float_ threshold, std::vector<Index_>* output_indices, std::vector<Float_>* output_distances) {
        auto normalized = internal::l2norm(ptr, buffer.size(), buffer.data());
        return my_searcher->search_all(normalized, threshold, output_indices, output_distances);
    }
    /**
     * @endcond
     */
};

/**
 * @brief Wrapper around a prebuilt index with L2 normalization.
 * 
 * This class's `unique_raw()` method creates a `Searcher` instance that L2-normalizes each query vector, typically for calculation of cosine distances.
 * Instances of this class are typically constructed with `L2NormalizedBuilder::unique_raw()`.
 *
 * @tparam Index_ Integer type for the indices.
 * For the output of `Builder::build`, this is set to `MockMatrix::index_type`.
 * @tparam Float_ Floating point type for the query data and output distances.
 */
template<typename Dim_, typename Index_, typename Float_>
class L2NormalizedPrebuilt : public Prebuilt<Dim_, Index_, Float_> {
public:
    /**
     * @param prebuilt Pointer to a `Prebuilt` instance for the neighbor search that is to be wrapped.
     */
    L2NormalizedPrebuilt(std::unique_ptr<Prebuilt<Dim_, Index_, Float_> > prebuilt) : my_prebuilt(std::move(prebuilt)) {}

private:
    std::unique_ptr<Prebuilt<Dim_, Index_, Float_> > my_prebuilt;

public:
    /**
     * @cond
     */
    Index_ num_observations() const {
        return my_prebuilt->num_observations();
    }

    Dim_ num_dimensions() const {
        return my_prebuilt->num_dimensions();
    }
    /**
     * @endcond
     */

    /**
     * Creates a `L2NormalizedSearcher` instance.
     */
    std::unique_ptr<Searcher<Index_, Float_> > initialize() const {
        return std::make_unique<L2NormalizedSearcher<Index_, Float_> >(my_prebuilt->initialize(), my_prebuilt->num_dimensions());
    }
};

/**
 * @brief Wrapper around a matrix with L2 normalization.
 *
 * @tparam Matrix_ Any class that satisfies the `MockMatrix_` interface.
 * 
 * This class satisfies the `MockMatrix` interface and performs L2 normalization of each observation's data vector in its implementation of `MockMatrix::get_observation()`.
 * It is mainly intended for use as a template argument when defining a `builder` for the `L2NormalizedBuilder` constructor.
 * In general, users should not be constructing an actual instance of this class.
 */
template<class Matrix_ = SimpleMatrix<int, int, double> >
class L2NormalizedMatrix {
/**
 * @cond
 */
public:
    L2NormalizedMatrix(const Matrix_& matrix) : my_matrix(matrix) {}

private:
    const Matrix_& my_matrix;

public:
    typedef typename Matrix_::data_type data_type;
    typedef typename Matrix_::index_type index_type;
    typedef typename Matrix_::dimension_type dimension_type;

    dimension_type num_dimensions() const {
        return my_matrix.num_dimensions();
    }

    index_type num_observations() const {
        return my_matrix.num_observations();
    }

    struct Workspace {
        Workspace(size_t n) : normalized(n) {}
        typename Matrix_::Workspace inner;
        std::vector<data_type> normalized;
    };

    Workspace create_workspace() const {
        return Workspace(my_matrix.num_dimensions());
    }

    const data_type* get_observation(Workspace& workspace) const {
        auto ptr = my_matrix.get_observation(workspace.inner);
        size_t ndim = workspace.normalized.size();
        return internal::l2norm(ptr, ndim, workspace.normalized.data());
    } 
/**
 * @endcond
 */
};

/**
 * @brief Wrapper around a builder with L2 normalization.
 *
 * This class applies L2 normalization to each observation vector in its input matrix, and also constructs `Searcher` instances that L2-normalize each query vector.
 * The premise is that Euclidean distances on L2-normalized vectors are monotonic transformations of the cosine distance.
 * Thus, given an arbitrary algorithm that finds nearest neighbors according to Euclidean distance, 
 * users can wrap the former's `Builder` with this `L2NormalizedBuilder` to obtain neighbors according to the cosine distance.
 */
template<class Matrix_ = SimpleMatrix<int, int, double>, typename Float_ = double>
class L2NormalizedBuilder : public Builder<Matrix_, Float_> {
public:
    /**
     * @param builder Pointer to a `Builder` for an arbitrary neighbor search algorithm.
     * This should be parametrized to accept an `L2NormalizedMatrix` wrapper around the intended matrix type.
     */
    L2NormalizedBuilder(std::unique_ptr<Builder<L2NormalizedMatrix<Matrix_>, Float_> > builder) : my_builder(std::move(builder)) {}

    /**
     * @param builder Pointer to a `Builder` for an arbitrary neighbor search algorithm.
     * This should be parametrized to accept an `L2NormalizedMatrix` wrapper around the intended matrix type.
     */
    L2NormalizedBuilder(Builder<L2NormalizedMatrix<Matrix_>, Float_>* builder) : my_builder(builder) {}

private:
    std::unique_ptr<Builder<L2NormalizedMatrix<Matrix_>, Float_> > my_builder;

public:
    /**
     * Creates a `L2NormalizedPrebuilt` instance.
     */
    Prebuilt<typename Matrix_::dimension_type, typename Matrix_::index_type, Float_>* build_raw(const Matrix_& data) const {
        return new L2NormalizedPrebuilt<typename Matrix_::dimension_type, typename Matrix_::index_type, Float_>(my_builder->build_unique(L2NormalizedMatrix(data)));
    }
};

}

#endif
