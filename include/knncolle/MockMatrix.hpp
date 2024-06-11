#ifndef KNNCOLLE_MOCK_MATRIX_HPP
#define KNNCOLLE_MOCK_MATRIX_HPP

/**
 * @file MockMatrix.hpp
 * @brief Expectations for matrix inputs.
 */

namespace knncolle {

/**
 * @brief Compile-time interface for matrix data.
 *
 * This defines the expectations for a matrix of observation-level data to be used in `Builder::build_raw()`.
 * Each matrix should support extraction of the vector of coordinates for each observation.
 */
class MockMatrix {
public:
    /**
     * @cond
     */
    MockMatrix(int num_dim, int num_obs, const double* data) : my_num_dim(num_dim), my_num_obs(num_obs), my_data(data), my_long_num_dim(num_dim) {}
    /**
     * @endcond
     */

public:
    /**
     * Type of the data.
     * Any floating-point type may be used here.
     */
    typedef double data_type;

    /**
     * Type for the observation indices.
     * Any integer type may be used here.
     */
    typedef int index_type;

    /**
     * Integer type for the dimension indices.
     * Any integer type may be used here.
     */
    typedef int dimension_type;

private:
    dimension_type my_num_dim;
    index_type my_num_obs;
    const data_type* my_data;
    size_t my_long_num_dim;

public:
    /**
     * @return Number of observations.
     */
    index_type num_observations() const {
        return my_num_obs;
    }

    /**
     * @return Number of dimensions.
     */
    dimension_type num_dimensions() const {
        return my_num_dim;
    }

public:
    /**
     * @brief Workspace for consecutive access to all observations.
     *
     * This should be used by matrix implementations to store temporary data structures that can be re-used in each call to `get_observation()`.
     * In particular, it should store the position of the current observation to be extracted by `get_observation()`,
     * which should then be incremented for subsequent calls.
     */
    struct Workspace {
        /**
         * @cond
         */
        size_t at = 0;
        /**
         * @endcond
         */
    };

    /**
     * @return A new consecutive-access workspace, to be passed to `get_observation()`.
     */
    Workspace create_workspace() const {
        return Workspace();
    }

    /**
     * @param workspace Workspace for consecutive access.
     * @return Pointer to an array of length equal to `num_dimensions()`, containing the coordinates for the next observation.
     *
     * For a newly created `Workspace`, the first call to `get_observation()` should return the coordinates of the first observation in the dataset;
     * the next call should return the coordinates of the second observation;
     * and so on, for up to `num_observations()` calls, after which the `Workspace` should no longer be used.
     */
    const data_type* get_observation(Workspace& workspace) const {
        return my_data + (workspace.at++) * my_long_num_dim; // already size_t to avoid overflow during multiplication.
    } 
};

/**
 * @brief Simple wrapper for an in-memory matrix.
 *
 * This defines a simple column-major matrix of observations where the columns are observations and the rows are dimensions.
 * It is compatible with the compile-time interface described in `MockMatrix`.
 *
 * @tparam Dim_ Integer type for the dimensions.
 * @tparam Index_ Integer type for the observation indices.
 * @tparam Data_ Floating-point type for the data.
 */
template<typename Dim_, typename Index_, typename Data_>
class SimpleMatrix {
public:
    /**
     * @param num_dimensions Number of dimensions.
     * @param num_observations Number of observations.
     * @param[in] data Pointer to an array of length `num_dim * num_obs`, containing a column-major matrix of observation data.
     * It is expected that the array will not be deallocated during the lifetime of this `SimpleMatrix` instance.
     */
    SimpleMatrix(Dim_ num_dimensions, Index_ num_observations, const Data_* data) : 
        my_num_dim(num_dimensions), my_num_obs(num_observations), my_data(data), my_long_num_dim(num_dimensions) {}

private:
    Dim_ my_num_dim;
    Index_ my_num_obs;
    const Data_* my_data;
    size_t my_long_num_dim;

public:
    /**
     * @cond
     */
    typedef Data_ data_type;

    typedef Index_ index_type;

    typedef Dim_ dimension_type;

    struct Workspace{
        size_t at = 0;
    };

public:
    Index_ num_observations() const {
        return my_num_obs;
    }

    dimension_type num_dimensions() const {
        return my_num_dim;
    }

    Workspace create_workspace() const {
        return Workspace();
    }

    const data_type* get_observation(Workspace& workspace) const {
        return my_data + (workspace.at++) * my_long_num_dim; // already size_t's to avoid overflow during multiplication.
    } 
    /**
     * @endcond
     */
};

}

#endif 
