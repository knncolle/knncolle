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

As one might expect, the name is not an accident.

<p float="left">
  <img src="https://i.makeagif.com/media/2-26-2015/JDQzgr.gif" width="32%" />
  <img src="https://thumbs.gfycat.com/SneakyPracticalIndianringneckparakeet-max-1mb.gif" width="32%" />
  <img src="https://media.tenor.com/images/2b3d5c70f6f4919320480f13427d881c/tenor.gif" width="32%" />
</p>

## Quick start

Given a matrix with dimensions in the rows and observations in the columns, we can do:

```cpp
#include "knncolle/knncolle.hpp"

/* ... boilerplate... */

knncolle::VpTreeEuclidean<> searcher(ndim, nobs, matrix.data()); 
auto results1 = searcher.find_nearest_neighbors(0, 10); // 10 nearest neighbors of the first element.
auto results2 = searcher.find_nearest_neighbors(query, 10); // 10 nearest neighbors of a query vector.
```

The `find_nearest_neighbors()` call will return a vector of (index, distance) pairs,
containing the requested number of neighbors in order of increasing distance from the query point.
(In cases where the requested number of neighbors is greater than the actual number of neighbors, the latter is returned.)
Each call is `const` and can be performed simultaneously in multiple threads, e.g., via OpenMP.

For some algorithms, we can modify the parameters of the search by passing our desired values in the constructor:

```cpp
knncolle::AnnoyEuclidean<> searcher2(ndim, nobs, matrix.data(), /* ntrees = */ 100); 
```

All algorithms derive from a common base class, so it is possible to swap algorithms at run-time:

```cpp
std::unique_ptr<knncolle::Base<> > ptr;
if (algorithm == "Annoy") {
    ptr.reset(new knncolle::AnnoyEuclidean<>(ndim, nobs, matrix.data()));
} else if (algorithm == "Hnsw") {
    ptr.reset(new knncolle::HnswEuclidean<>(ndim, nobs, matrix.data()));
} else {
    ptr.reset(new knncolle::KmknnEuclidean<>(ndim, nobs, matrix.data()));
}
auto res = ptr->find_nearest_neighbors(1, 10);
```

Each class is also templated, defaulting to `int`s for the indices and `double`s for the distances.
If precision is not a concern, one can often achieve greater speed by swapping all `double`s with `float`s.

Check out the [reference documentation](https://ltla.github.io/knncolle/) for more details.

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

## Further comments

Both Annoy and HNSW will attempt to perform manual vectorization based on SSE and/or AVX instructions.
This may result in differences in the results across machines due to changes in numeric precision across architectures with varying support for SSE/AVX intrinsics.
For the most part, such differences can be avoided by consistently compiling for the "near-lowest common denominator" (such as the typical `x86-64` default for GCC and Clang) 
and ensuring that the more specific instruction subsets like SSE3 and AVX are not enabled (which are typically off by default anyway).
Nonetheless, if reproducibility across architectures is important, it can be achieved at the cost of some speed by defining the `NO_MANUAL_VECTORIZATION` macro,
which will instruct both Annoy and HNSW to disable their vectorized optimizations.

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

