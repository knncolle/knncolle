#ifndef KNNCOLLE_L2_NORMALIZED_HP
#define KNNCOLLE_L2_NORMALIZED_HP

#include <vector>
#include <cmath>
#include <memory>
#include <limits>

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

template<typename Data_, typename Normalized_>
void l2norm(const Data_* ptr, size_t ndim, Normalized_* buffer) {
    Normalized_ l2 = 0;
    for (size_t d = 0; d < ndim; ++d) {
        Normalized_ val = ptr[d]; // cast to Normalized_ to avoid issues with integer overflow.
        buffer[d] = val;
        l2 += val * val;
    }

    if (l2 > 0) {
        l2 = std::sqrt(l2);
        for (size_t d = 0; d < ndim; ++d) {
            buffer[d] /= l2;
        }
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
 * @tparam Dim_ Integer type for the number of dimensions.
 * @tparam Index_ Integer type for the indices.
 * @tparam Data_ Floating-point type for the input and query data.
 * @tparam Distance_ Floating-point type for the distances.
 * @tparam Normalized_ Floating-point type for the L2-normalized data.
 */
template<typename Dim_, typename Index_, typename Data_, typename Distance_, typename Normalized_>
class L2NormalizedSearcher final : public Searcher<Index_, Data_, Distance_> {
public:
    /**
     * @param searcher Pointer to a `Searcher` class for the neighbor search that is to be wrapped.
     * @param num_dimensions Number of dimensions of the data.
     */
    L2NormalizedSearcher(std::unique_ptr<Searcher<Index_, Float_> > searcher, Dim_ num_dimensions) : 
        my_searcher(std::move(searcher)),
        buffer(num_dimensions)
    {}

private:
    // No way around this; the L2-normalized values must be floating-point,
    // so the internal searcher must accept floats.
    static_assert(std::is_floating_point<Normalized_>::value);

    std::unique_ptr<Searcher<Index_, Normalized_, Distance_> > my_searcher;
    std::vector<Normalized_> buffer;
    /**
     * @cond
     */

public:
    void search(Index_ i, Index_ k, std::vector<Index_>* output_indices, std::vector<Float_>* output_distances) {
        my_searcher->search(i, k, output_indices, output_distances);
    }

    void search(const Data_* ptr, Index_ k, std::vector<Index_>* output_indices, std::vector<Float_>* output_distances) {
        auto normalized = buffer.data();
        internal::l2norm(ptr, buffer.size(), normalized);
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
        auto normalized = buffer.data();
        internal::l2norm(ptr, buffer.size(), normalized);
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
 * @tparam Dim_ Integer type for the number of dimensions.
 * @tparam Index_ Integer type for the indices.
 * @tparam Data_ Floating-point type for the input and query data.
 * @tparam Distance_ Floating-point type for the distances.
 * @tparam Normalized_ Floating-point type for the L2-normalized data.
 */
template<typename Dim_, typename Index_, typename Data_, typename Distance_, typename Normalized_>
class L2NormalizedPrebuilt final : public Prebuilt<Dim_, Index_, Data_, Distance_> {
public:
    /**
     * @param prebuilt Pointer to a `Prebuilt` instance for the neighbor search that is to be wrapped.
     */
    L2NormalizedPrebuilt(std::unique_ptr<Prebuilt<Dim_, Index_, Normalized_, Distance_> > prebuilt) : my_prebuilt(std::move(prebuilt)) {}

private:
    std::unique_ptr<Prebuilt<Dim_, Index_, Normalized_, Distance_> > my_prebuilt;

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
    std::unique_ptr<Searcher<Index_, Data_, Distance_> > initialize() const {
        return std::make_unique<L2NormalizedSearcher<Dim_, Index_, Data_, Distance_, Normalized_> >(my_prebuilt->initialize(), my_prebuilt->num_dimensions());
    }
};

/**
 * @cond
 */
template<typename Dim_, typename Index_, typename Data_, typename Normalized_, typename Matrix_ = Matrix<Dim_, Index_, Data_> >
class L2NormalizedMatrix final : public Matrix<Dim_, Index_, Normalized_>;
/**
 * @endcond
 */

/**
 * @brief Extractor for the `L2NormalizedMatrix`.
 *
 * @tparam Dim_ Integer type for the number of dimensions.
 * @tparam Index_ Integer type for the indices.
 * @tparam Data_ Numeric type for the original matrix data.
 * @tparam Normalized_ Floating-point type for the L2-normalized data.
 */
template<typename Dim_, typename Index_, typename Data_, typename Normalized_>
class L2NormalizedMatrixExtractor final : public MatrixExtractor<Dim_, Index_, Normalized_> {
public:
    /**
     * @cond
     */
    L2NormalizedMatrixExtractor(std::unique_ptr<MatrixExtractor<Dim_, Index_, Data_> > extractor, Dim_ dim) : 
        my_extractor(std::move(extractor)), buffer(dim) {}

private:
    std::unique_ptr<MatrixExtractor<Dim_, Index_, Data_> > my_extractor;
    std::vector<Normalized_> buffer;

public:
    const Normalized_* next() {
        auto raw = my_extractor->next();
        auto normalized = buffer.data();
        internal::l2norm(raw, buffer.size(), normalized);
        return normalized;
    }
    /**
     * @endcond
     */
};

/**
 * @brief Wrapper around a matrix with L2 normalization.
 *
 * @tparam Dim_ Integer type for the number of dimensions.
 * @tparam Index_ Integer type for the indices.
 * @tparam Data_ Numeric type for the original matrix data.
 * @tparam Normalized_ Floating-point type for the L2-normalized data.
 * @tparam Matrix_ Class that satisfies the `Matrix` interface.
 * 
 * This class satisfies the `Matrix` interface and performs L2 normalization of each observation's data vector in its implementation of `MatrixExtractor::next()`.
 * It is mainly intended for use as a template argument when defining a `builder` for the `L2NormalizedBuilder` constructor.
 * In general, users should not be constructing an actual instance of this class.
 */
template<typename Dim_, typename Index_, typename Data_, typename Normalized_, typename Matrix_ = Matrix<Dim_, Index_, Data_> >
class L2NormalizedMatrix final : public Matrix<Dim_, Index_, Normalized_> {
/**
 * @cond
 */
public:
    L2NormalizedMatrix(const Matrix_& matrix) : my_matrix(matrix) {}

private:
    const Matrix_& my_matrix;

public:
    Dim_ num_dimensions() const {
        return my_matrix.num_dimensions();
    }

    Index_ num_observations() const {
        return my_matrix.num_observations();
    }

    std::unique_ptr<MatrixExtractor<Data_> > new_extractor() const {
        return std::make_unique<L2NormalizedMatrixExtractor<Dim_, Index_, Data_, Normalized_> >(my_matrix.new_extractor(), num_dimensions());
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
 *
 * @tparam Dim_ Integer type for the number of dimensions.
 * @tparam Index_ Integer type for the indices.
 * @tparam Data_ Numeric type for the input and query data.
 * @tparam Distance_ Floating point type for the distances.
 * @tparam Normalized_ Floating-point type for the L2-normalized data.
 * @tparam Matrix_ The `Matrix` interface class. 
 * This is restricted as it must be compatible with both the `L2NormalizedMatrix` and the type of the original matrix.
 * It is only parametrized here for consistency with the other `Builder` classes, and the default should not be changed.
 */
template<typename Dim_, typename Index_, typename Data_, typename Distance_, typename Normalized_, class Matrix_ = Matrix<Dim_, Index_, Data_> >
class L2NormalizedBuilder final : public Builder<Dim_, Index_, Data_, Distance_, Matrix_> {
public:
    /**
     * @param builder Pointer to a `Builder` for an arbitrary neighbor search algorithm.
     */
    L2NormalizedBuilder(std::unique_ptr<const Builder<Dim_, Index_, Data_, Distance_, Matrix_> > builder) : my_builder(std::move(builder)) {}

    /**
     * @param builder Pointer to a `Builder` for an arbitrary neighbor search algorithm.
     */
    L2NormalizedBuilder(const Builder<Dim_, Index_, Data_, Distance_, Matrix_>* builder) : my_builder(builder) {}

private:
    std::unique_ptr<const Builder<L2NormalizedMatrix<Matrix_>, Float_> > my_builder;

public:
    /**
     * Creates a `L2NormalizedPrebuilt` instance.
     */
    Prebuilt<Dim_, Index_, Data_, Distance_>* build_raw(const Matrix_& data) const {
        L2NormalizedMatrix<Dim_, Index_, Data_, Normalized_, Matrix_> normalized(data)
        return new L2NormalizedPrebuilt<Dim_, Index_, Data_, Distance_, Normalized_>(my_builder->build_unique(normalized));
    }
};

}

#endif
