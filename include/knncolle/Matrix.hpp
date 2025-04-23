#ifndef KNNCOLLE_MATRIX_HPP
#define KNNCOLLE_MATRIX_HPP

#include <memory>
#include <cstddef>

/**
 * @file Matrix.hpp
 * @brief Interface for the input matrix.
 */

namespace knncolle {

/**
 * @brief Extractor interface for matrix data.
 *
 * @tparam Data_ Numeric type of the data.
 */
template<typename Data_>
class MatrixExtractor {
public:
    /**
     * @cond
     */
    MatrixExtractor() = default;
    MatrixExtractor(const MatrixExtractor&) = default;
    MatrixExtractor(MatrixExtractor&&) = default;
    MatrixExtractor& operator=(const MatrixExtractor&) = default;
    MatrixExtractor& operator=(MatrixExtractor&&) = default;
    virtual ~MatrixExtractor() = default;
    /**
     * @endcond
     */

public:
    /**
     * @return Pointer to an array of length equal to `Matrix::num_dimensions()`, containing the coordinates for the next observation.
     *
     * For a newly created `MatrixExtractor`, the first call to `next()` should return the coordinates of the first observation in the matrix;
     * the next call should return the coordinates of the second observation;
     * and so on, for up to `Matrix::num_observations()` calls, after which the `MatrixExtractor` should no longer be used.
     */
    virtual const Data_* next() = 0;
};


/**
 * @brief Interface for matrix data.
 *
 * This defines the expectations for a matrix of observation-level data to be used in `Builder::build_raw()`.
 *
 * @tparam Index_ Integer type of the observation indices.
 * @tparam Data_ Numeric type of the data.
 */
template<typename Index_, typename Data_>
class Matrix {
public:
    /**
     * @cond
     */
    Matrix() = default;
    Matrix(const Matrix&) = default;
    Matrix(Matrix&&) = default;
    Matrix& operator=(const Matrix&) = default;
    Matrix& operator=(Matrix&&) = default;
    virtual ~Matrix() = default;
    /**
     * @endcond
     */

public:
    /**
     * @return Number of observations.
     */
    virtual Index_ num_observations() const = 0;

    /**
     * @return Number of dimensions.
     */
    virtual std::size_t num_dimensions() const = 0;

public:
    /**
     * @return A new consecutive-access extractor.
     */
    virtual std::unique_ptr<MatrixExtractor<Data_> > new_extractor() const = 0;
};

/**
 * @brief Extractor for a `SimpleMatrix`.
 *
 * This should be typically constructed by calling `SimpleMatrix::new_extractor()`.
 *
 * @tparam Data_ Numeric type of the data.
 */
template<typename Data_>
class SimpleMatrixExtractor final : public MatrixExtractor<Data_> {
public:
    /**
     * @cond
     */
    SimpleMatrixExtractor(const Data_* data, std::size_t dim) : my_data(data), my_dim(dim) {}
    /**
     * @endcond
     */

private:
    const Data_* my_data;
    std::size_t my_dim;
    std::size_t at = 0;

public:
    const Data_* next() {
        return my_data + (at++) * my_dim; // already std::size_t's to avoid overflow during multiplication.
    } 
};

/**
 * @brief Simple wrapper for an in-memory matrix.
 *
 * This defines a simple column-major matrix of observations where the columns are observations and the rows are dimensions.
 * It is compatible with the compile-time interface described in `MockMatrix`.
 *
 * @tparam Index_ Integer type of the observation indices.
 * @tparam Data_ Numeric type of the data.
 */
template<typename Index_, typename Data_>
class SimpleMatrix final : public Matrix<Index_, Data_> {
public:
    /**
     * @param num_dimensions Number of dimensions.
     * @param num_observations Number of observations.
     * @param[in] data Pointer to an array of length `num_dim * num_obs`, containing a column-major matrix of observation data.
     * It is expected that the array will not be deallocated during the lifetime of this `SimpleMatrix` instance.
     */
    SimpleMatrix(std::size_t num_dimensions, Index_ num_observations, const Data_* data) : 
        my_num_dim(num_dimensions), my_num_obs(num_observations), my_data(data) {}

private:
    std::size_t my_num_dim;
    Index_ my_num_obs;
    const Data_* my_data;

public:
    Index_ num_observations() const {
        return my_num_obs;
    }

    std::size_t num_dimensions() const {
        return my_num_dim;
    }

    std::unique_ptr<MatrixExtractor<Data_> > new_extractor() const {
        return std::make_unique<SimpleMatrixExtractor<Data_> >(my_data, my_num_dim);
    }
};

}

#endif 
