# Collection of KNN algorithms

![Unit tests](https://github.com/LTLA/knncolle/actions/workflows/run-tests.yaml/badge.svg)
![Documentation](https://github.com/LTLA/knncolle/actions/workflows/doxygenate.yaml/badge.svg)
[![Codecov](https://codecov.io/gh/LTLA/knncolle/branch/master/graph/badge.svg?token=PPKVZ1QEXR)](https://codecov.io/gh/LTLA/knncolle)

## Overview

**knncolle** is a header-only C++ library that collects a variety of different k-nearest neighbor algorithms under a consistent interface.
The aim is to enable downstream libraries to easily switch between different methods with a single runtime flag,
or by just swapping out the relevant constructors at compile time.

Currently, we support the following methods:

- [K-means with k-nearest neighbors](https://pubmed.ncbi.nlm.nih.gov/22247818/), an exact search that uses k-means clustering to index points.
- [Vantage point tree](http://stevehanov.ca/blog/?id=130), an exact search that uses the tree of the same name.
- [Annoy](https://github.com/spotify/annoy/), an approximate search based on random projections.
- [HNSW](https://github.com/nmslib/hnswlib/), an approximate search based on hierarchical graphs.
- Brute force search.

Most of the code in this library is derived from the [**BiocNeighbors** R package](https://bioconductor.org/packages/release/bioc/html/BiocNeighbors.html).

## Quick start

Given a matrix with dimensions in the rows and observations in the columns, we can do:

```cpp
#include "knncolle/knncolle.hpp"

// Wrap our data in a light SimpleMatrix.
knncolle::SimpleMatrix<int, int, double> mat(ndim, nobs, matrix.data());

// Build a VP-tree index. 
knncolle::VptreeBuilder<> vp_builder;
auto vp_index = vp_builder.build(mat);

// Find 10 nearest neighbors of every element.
auto results = knncolle::find_nearest_neighbors(*vp_index, 10); 
```

The `find_nearest_neighbors()` call will return a vector of (index, distance) pairs,
containing the requested number of neighbors in order of increasing distance from the query point.
(In cases where the requested number of neighbors is greater than the actual number of neighbors, the latter is returned.)

Check out the [reference documentation](https://ltla.github.io/knncolle/) for more details.

## Searching in more detail

We can perform the search manually by constructing a `Searcher` instance and looping over the elements of interest.
Continuing with the same variables defined in the previous section, we could replace the `find_nearest_neighbors()` call with:

```cpp
auto searcher = vp_index->initialize();
std::vector<std::pair<int, double> > results;
for (int o = 0; o < nobs; ++o) {
    searcher->search(o, 10, results);
    // Do something with the search 'results' for 'o'.
}
```

Similarly, we can query the prebuilt index for the neighbors of an arbitrary vector.
The code below searches for the nearest 5 neighbors to a query vector at the origin:

```cpp
std::vector<double> query(ndim);
searcher->search(query.data(), 5, results);
```

To parallelize the loop, we just need to construct a separate `Searcher` (and the result vector) for each thread.
This is already implemented in `find_nearest_neighbors()` but is also easy to do by hand, e.g., with OpenMP:

```cpp
#pragma omp parallel num_threads(5)
{
    auto searcher = vp_index->initialize();
    std::vector<std::pair<int, double> > results;
    #pragma omp for
    for (int o = 0; o < nobs; ++o) {
        searcher->search(o, 10, results);
        // Do something with the search 'results' for 'o'.
    }
}
```

## Tuning index construction

Some algorithms allow the user to modify the parameters of the search by passing options in the relevant `Builder` constructor.
For example, the KMKNN method has several options for the k-means clustering step.
We could, say, specify which initialization algorithm to use:

```cpp
knncolle::KmknnOptions<> kk_opt;
kk_opt.initialize_algorithm.reset(
    new kmeans::InitializeRandom<kmeans::SimpleMatrix<double, int, int>, int, double>
);
```

Or modify the behavior of the refinement algorithm:

```cpp
kmeans::RefineLloydOptions ll_opt;
ll_opt.max_iterations = 20;
ll_opt.num_threads = 5;
kk_opt.refine_algorithm.reset(
    new kmeans::RefineLloyd<kmeans::SimpleMatrix<double, int, int>, int, double>(ll_opt)
);
```

After which, we construct our `KmknnBuilder`, build our `KmknnPrebuilt` index, and proceed with the nearest-neighbor search.

```cpp
knncolle::KmknnBuilder<> kk_builder(kk_opt);
auto kk_prebuilt = kk_builder.build(mat);
auto kk_results = knncolle::find_nearest_neighbors(*kk_prebuilt, 10); 
```

Check out the [reference documentation](https://ltla.github.io/knncolle/) for the available options in each algorithm's `Builder`.

## Polymorphism

All methods implement the `Builder`, `Prebuilt` and `Searcher` interfaces via inheritance.
This means that users can swap algorithms at run-time:

```cpp
std::unique_ptr<knncolle::Builder<> > ptr;
if (algorithm == "brute-force") {
    ptr.reset(new knncolle::BruteforceBuilder<>);
} else if (algorithm == "kmknn") {
    ptr.reset(new knncolle::KmknnBuilder<>);
} else {
    ptr.reset(new knncolle::VptreeBuilder<>);
}

auto some_prebuilt = ptr->build(mat);
auto some_results = knncolle::find_nearest_neighbors(*some_prebuilt, 10); 
```

Each class is also heavily templated to enable compile-time polymorphism:

- We default to `int`s for the indices and `double`s for the distances.
  If precision is not a concern, one can often achieve greater speed by swapping all `double`s with `float`s.
- The choice of distance calculation is often a compile-time parameter, as defined by the `MockDistance` compile-time interface.
  Advanced users can define their own classes to customize the distance calculations.
- The choice of input data is another compile-time paramter, as defined by the `MockMatrix` interface.
  Advanced users can define their own inputs to, e.g., read from file-backed or sparse matrices.

Check out the [reference documentation](https://ltla.github.io/knncolle/) for more details on these interfaces.

## Building projects with **knncolle**

### CMake with `FetchContent`

If you're using CMake, you just need to add something like this to your `CMakeLists.txt`:

```cmake
include(FetchContent)

FetchContent_Declare(
  knncolle
  GIT_REPOSITORY https://github.com/LTLA/knncolle
  GIT_TAG master # or any version of interest
)

FetchContent_MakeAvailable(knncolle)
```

Then you can link to **knncolle** to make the headers available during compilation:

```cmake
# For executables:
target_link_libraries(myexe ltla::knncolle)

# For libaries
target_link_libraries(mylib INTERFACE ltla::knncolle)
```

### CMake with `find_package()`

```cmake
find_package(ltla_knncolle CONFIG REQUIRED)
target_link_libraries(mylib INTERFACE ltla::knncolle)
```

To install the library, use:

```sh
mkdir build && cd build
cmake .. -DKNNCOLLE_TESTS=OFF
cmake --build . --target install
```

By default, this will use `FetchContent` to fetch all external dependencies.
If you want to install them manually, use `-DKNNCOLLE_FETCH_EXTERN=OFF`.
See the commit hashes in [`extern/CMakeLists.txt`](extern/CMakeLists.txt) to find compatible versions of each dependency.

### Manual

If you're not using CMake, the simple approach is to just copy the files in `include/` - either directly or with Git submodules - and include their path during compilation with, e.g., GCC's `-I`.
This requires the external dependencies listed in [`extern/CMakeLists.txt`](extern/CMakeLists.txt), which also need to be made available during compilation.

## References

Wang X (2012). 
A fast exact k-nearest neighbors algorithm for high dimensional search using k-means clustering and triangle inequality. 
_Proc Int Jt Conf Neural Netw_, 43, 6:2351-2358.

Hanov S (2011).
VP trees: A data structure for finding stuff fast.
http://stevehanov.ca/blog/index.php?id=130

Yianilos PN (1993).
Data structures and algorithms for nearest neighbor search in general metric spaces.
_Proceedings of the Fourth Annual ACM-SIAM Symposium on Discrete Algorithms_, 311-321.

Bernhardsson E (2018).
Annoy.
https://github.com/spotify/annoy

Malkov YA, Yashunin DA (2016).
Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs.
_arXiv_,
https://arxiv.org/abs/1603.09320

