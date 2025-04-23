#ifndef KNNCOLLE_L2_NORMALIZED_HP
#define KNNCOLLE_L2_NORMALIZED_HP

#include <vector>
#include <cmath>
#include <memory>
#include <limits>
#include <cstddef>

#include "Searcher.hpp"
#include "Prebuilt.hpp"
#include "Builder.hpp"
#include "Matrix.hpp"

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
void l2norm(const Data_* ptr, std::size_t ndim, Normalized_* buffer) {
    Normalized_ l2 = 0;
    for (std::size_t d = 0; d < ndim; ++d) {
        Normalized_ val = ptr[d]; // cast to Normalized_ to avoid issues with integer overflow.
        buffer[d] = val;
        l2 += val * val;
    }

    if (l2 > 0) {
        l2 = std::sqrt(l2);
        for (std::size_t d = 0; d < ndim; ++d) {
            buffer[d] /= l2;
        }
    }
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
 * @tparam Data_ Numeric type for the input and query data.
 * @tparam Distance_ Floating-point type for the distances.
 * @tparam Normalized_ Floating-point type for the L2-normalized data.
 */
template<typename Index_, typename Data_, typename Distance_, typename Normalized_>
class L2NormalizedSearcher final : public Searcher<Index_, Data_, Distance_> {
public:
    /**
     * @param searcher Pointer to a `Searcher` class for the neighbor search that is to be wrapped.
     * @param num_dimensions Number of dimensions of the data.
     */
    L2NormalizedSearcher(std::unique_ptr<Searcher<Index_, Normalized_, Distance_> > searcher, std::size_t num_dimensions) : 
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
    void search(Index_ i, Index_ k, std::vector<Index_>* output_indices, std::vector<Distance_>* output_distances) {
        my_searcher->search(i, k, output_indices, output_distances);
    }

    void search(const Data_* ptr, Index_ k, std::vector<Index_>* output_indices, std::vector<Distance_>* output_distances) {
        auto normalized = buffer.data();
        internal::l2norm(ptr, buffer.size(), normalized);
        my_searcher->search(normalized, k, output_indices, output_distances);
    }

public:
    bool can_search_all() const {
        return my_searcher->can_search_all();
    }

    Index_ search_all(Index_ i, Distance_ threshold, std::vector<Index_>* output_indices, std::vector<Distance_>* output_distances) {
        return my_searcher->search_all(i, threshold, output_indices, output_distances);
    }

    Index_ search_all(const Data_* ptr, Distance_ threshold, std::vector<Index_>* output_indices, std::vector<Distance_>* output_distances) {
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
 * @tparam Index_ Integer type for the indices.
 * @tparam Data_ Floating-point type for the input and query data.
 * @tparam Distance_ Floating-point type for the distances.
 * @tparam Normalized_ Floating-point type for the L2-normalized data.
 */
template<typename Index_, typename Data_, typename Distance_, typename Normalized_>
class L2NormalizedPrebuilt final : public Prebuilt<Index_, Data_, Distance_> {
public:
    /**
     * @param prebuilt Pointer to a `Prebuilt` instance for the neighbor search that is to be wrapped.
     */
    L2NormalizedPrebuilt(std::unique_ptr<Prebuilt<Index_, Normalized_, Distance_> > prebuilt) : my_prebuilt(std::move(prebuilt)) {}

private:
    std::unique_ptr<Prebuilt<Index_, Normalized_, Distance_> > my_prebuilt;

public:
    /**
     * @cond
     */
    Index_ num_observations() const {
        return my_prebuilt->num_observations();
    }

    std::size_t num_dimensions() const {
        return my_prebuilt->num_dimensions();
    }
    /**
     * @endcond
     */

    /**
     * Creates a `L2NormalizedSearcher` instance.
     */
    std::unique_ptr<Searcher<Index_, Data_, Distance_> > initialize() const {
        return std::make_unique<L2NormalizedSearcher<Index_, Data_, Distance_, Normalized_> >(my_prebuilt->initialize(), my_prebuilt->num_dimensions());
    }
};

/**
 * @cond
 */
template<typename Index_, typename Data_, typename Normalized_, typename Matrix_>
class L2NormalizedMatrix;
/**
 * @endcond
 */

/**
 * @brief Extractor for the `L2NormalizedMatrix`.
 *
 * @tparam Index_ Integer type for the indices.
 * @tparam Data_ Numeric type for the original matrix data.
 * @tparam Normalized_ Floating-point type for the L2-normalized data.
 */
template<typename Index_, typename Data_, typename Normalized_>
class L2NormalizedMatrixExtractor final : public MatrixExtractor<Normalized_> {
public:
    /**
     * @cond
     */
    L2NormalizedMatrixExtractor(std::unique_ptr<MatrixExtractor<Data_> > extractor, std::size_t dim) : 
        my_extractor(std::move(extractor)), buffer(dim) {}

private:
    std::unique_ptr<MatrixExtractor<Data_> > my_extractor;
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
 * @tparam Index_ Integer type for the indices.
 * @tparam Data_ Numeric type for the original matrix data.
 * @tparam Normalized_ Floating-point type for the L2-normalized data.
 * @tparam Matrix_ Class of the input data matrix. 
 * This should satisfy the `Matrix` interface.
 * 
 * This class satisfies the `Matrix` interface and performs L2 normalization of each observation's data vector in its implementation of `MatrixExtractor::next()`.
 * It is mainly intended for use as a template argument when defining a `builder` for the `L2NormalizedBuilder` constructor.
 * In general, users should not be constructing an actual instance of this class.
 */
template<typename Index_, typename Data_, typename Normalized_, typename Matrix_ = Matrix<Index_, Data_> >
class L2NormalizedMatrix final : public Matrix<Index_, Normalized_> {
/**
 * @cond
 */
public:
    L2NormalizedMatrix(const Matrix_& matrix) : my_matrix(matrix) {}

private:
    static_assert(std::is_same<decltype(std::declval<Matrix_>().num_observations()), Index_>::value);
    static_assert(std::is_same<typename std::remove_pointer<decltype(std::declval<Matrix_>().new_extractor()->next())>::type, const Data_>::value);

    const Matrix_& my_matrix;

public:
    std::size_t num_dimensions() const {
        return my_matrix.num_dimensions();
    }

    Index_ num_observations() const {
        return my_matrix.num_observations();
    }

    std::unique_ptr<MatrixExtractor<Normalized_> > new_extractor() const {
        return std::make_unique<L2NormalizedMatrixExtractor<Index_, Data_, Normalized_> >(my_matrix.new_extractor(), num_dimensions());
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
 * @tparam Index_ Integer type for the indices.
 * @tparam Data_ Numeric type for the input and query data.
 * @tparam Distance_ Floating point type for the distances.
 * @tparam Normalized_ Floating-point type for the L2-normalized data.
 * @tparam Matrix_ Class of the input data matrix. 
 * This should satisfy the `Matrix` interface.
 */
template<typename Index_, typename Data_, typename Distance_, typename Normalized_, class Matrix_ = Matrix<Index_, Data_> >
class L2NormalizedBuilder final : public Builder<Index_, Data_, Distance_, Matrix_> {
public:
    /**
     * Alias for the type of the normalized matrix.
     */
    typedef L2NormalizedMatrix<Index_, Data_, Normalized_, Matrix_> NormalizedMatrix;

    /**
     * The expected matrix type in the `Builder` instance in the constructor.
     * 
     * If `Matrix_` is a base class of `NormalizedMatrix`, the `Builder` can use the same `Matrix_` in its template parametrization.
     * This is because `Matrix_` will be compatible with both the `L2NormalizedMatrix` and the type of the original matrix,
     * allowing us to transparently substitute an instance of the latter with that of the former.
     * In general, this scenario requires the default `Matrix_` type and `Data_ == Normalized_`.
     *
     * Otherwise, the `Builder` should explicitly have a `NormalizedMatrix` in its template parametrization.
     * This is because it will be accepting the normalized matrix in `build_raw()` instead of `Matrix_`.
     */
    typedef typename std::conditional<
        std::is_base_of<Matrix_, NormalizedMatrix>::value,
        Matrix_,
        NormalizedMatrix
    >::type BuilderMatrix;

public:
    /**
     * @param builder Pointer to a `Builder` for an arbitrary neighbor search algorithm.
     */
    L2NormalizedBuilder(std::shared_ptr<const Builder<Index_, Normalized_, Distance_, BuilderMatrix> > builder) : my_builder(std::move(builder)) {}

private:
    std::shared_ptr<const Builder<Index_, Normalized_, Distance_, BuilderMatrix> > my_builder;

public:
    /**
     * Creates a `L2NormalizedPrebuilt` instance.
     */
    Prebuilt<Index_, Data_, Distance_>* build_raw(const Matrix_& data) const {
        NormalizedMatrix normalized(data);
        return new L2NormalizedPrebuilt<Index_, Data_, Distance_, Normalized_>(my_builder->build_unique(normalized));
    }
};

}

#endif
