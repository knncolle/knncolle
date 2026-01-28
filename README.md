# Collection of KNN algorithms

![Unit tests](https://github.com/knncolle/knncolle/actions/workflows/run-tests.yaml/badge.svg)
![Documentation](https://github.com/knncolle/knncolle/actions/workflows/doxygenate.yaml/badge.svg)
[![Codecov](https://codecov.io/gh/knncolle/knncolle/branch/master/graph/badge.svg)](https://codecov.io/gh/knncolle/knncolle)

## Overview

**knncolle** is a header-only C++ library that collects a variety of different k-nearest neighbor algorithms under a consistent interface.
The aim is to enable downstream libraries to easily switch between different methods with a single runtime flag,
or by just swapping out the relevant constructors at compile time.

The core library implements various interfaces along with the following methods:

- [Vantage point tree](http://stevehanov.ca/blog/?id=130), an exact search that uses the tree of the same name.
- Brute force search, mostly implemented for testing.

Additional libraries extend the **knncolle** framework to more algorithms:

- [**knncolle_kmknn**](https://github.com/knncolle/knncolle_kmknn) wraps [KMKNN](https://pubmed.ncbi.nlm.nih.gov/22247818/), an exact search based on k-means clustering.
- [**knncolle_annoy**](https://github.com/knncolle/knncolle_annoy) wraps [Annoy](https://github.com/spotify/annoy/), an approximate search based on random projections.
- [**knncolle_hnsw**](https://github.com/knncolle/knncolle_hnsw) wraps [HNSW](https://github.com/nmslib/hnswlib/), an approximate search based on hierarchical graphs.

Most of the code in this library is derived from the [**BiocNeighbors** R package](https://bioconductor.org/packages/release/bioc/html/BiocNeighbors.html).

## Quick start

Given a matrix with dimensions in the rows and observations in the columns, we can do:

```cpp
#include "knncolle/knncolle.hpp"

int ndim = 10;
int nobs = 1000;
std::vector<double> matrix(ndim * nobs); // column-major dims x obs matrix. 

// Wrap our data in a SimpleMatrix.
knncolle::SimpleMatrix<
    /* observation index */ int,
    /* data type */ double
> mat(ndim, nobs, matrix.data());

// Build a VP-tree index with double-precision Euclidean distances.
auto edist = std::make_shared<knncolle::EuclideanDistance<
    /* data type = */ double,
    /* distance type = */ double
> >();
knncolle::VptreeBuilder<
    /* observation index */ int, 
    /* data type */ double, 
    /* distance type */ double
> vp_builder(std::move(edist));
auto vp_index = vp_builder.build_unique(mat);

// Find 10 nearest neighbors of every observation.
auto results = knncolle::find_nearest_neighbors(*vp_index, 10); 

results[0].first; // indices of neighbors of the first observation
results[0].second; // distances to neighbors of the first observation
```

Check out the [reference documentation](https://knncolle.github.io/knncolle/) for more details.

## Searching in more detail

We can perform the search manually by constructing a `Searcher` instance and looping over the elements of interest.
Continuing with the same variables defined in the previous section, we could replace the `find_nearest_neighbors()` call with:

```cpp
auto searcher = vp_index->initialize();
std::vector<int> indices;
std::vector<double> distances;
for (int o = 0; o < nobs; ++o) {
    searcher->search(o, 10, &indices, &distances);
    // Do something with the search results for 'o'.
}
```

Similarly, we can query the prebuilt index for the neighbors of an arbitrary vector.
The code below searches for the nearest 5 neighbors to a query vector at the origin:

```cpp
std::vector<double> query(ndim);
searcher->search(query.data(), 5, &indices, &distances);
```

To parallelize the loop, we just need to construct a separate `Searcher` (and the result vector) for each thread.
This is already implemented in `find_nearest_neighbors()` but is also easy to do by hand, e.g., with OpenMP:

```cpp
#pragma omp parallel num_threads(5)
{
    auto searcher = vp_index->initialize();
    std::vector<int> indices;
    std::vector<double> distances;
    #pragma omp for
    for (int o = 0; o < nobs; ++o) {
        searcher->search(o, 10, &indices, &distances);
        // Do something with the search 'results' for 'o'.
    }
}
```

Either (or both) of `indices` and `distances` may be `NULL`, in which case the corresponding values are not reported.
This allows implementations to skip the extraction of distances when only the identities of the neighbors are of interest.

```cpp
searcher->search(0, 5, &indices, NULL);
```

## Finding all neighbors within range

A related problem involves finding all neighbors within a certain distance of an observation.
This can be achieved using the `Searcher::search_all()` method:

```cpp
if (seacher->can_search_all()) {
    // Report all neighbors within a distance of 10 from the first point.
    searcher->search_all(0, 10, &indices, &distances);

    // Report all neighbors within a distance of 0.5 from a query point.
    searcher->search_all(query.data(), 0.5, &indices, &distances);
}
```

This method is optional so developers of `Searcher` subclasses may choose to not implement it.
Applications should check `Searcher::can_search_all()` before attempting a call, as shown above.
Otherwise, the default method will raise an exception. 

## Polymoprhism via interfaces

All KNN search algorithms implement the `Builder`, `Prebuilt` and `Searcher` interfaces via inheritance.
This means that users can swap algorithms at run-time:

```cpp
auto dist_type = std::make_shared<knncolle::EuclideanDistance<double, double> >();

std::unique_ptr<knncolle::Builder<int, double, double> > ptr;
if (algorithm == "brute-force") {
    ptr.reset(new knncolle::BruteforceBuilder<int, double, double>(dist_type));
} else if (algorithm == "vp-tree") {
    ptr.reset(new knncolle::VptreeBuilder<int, double, double>(dist_type));
} else {
    // do something else
}

auto some_prebuilt = ptr->build_unique(mat);
auto some_results = knncolle::find_nearest_neighbors(*some_prebuilt, 10); 
```

Similarly, for algorithms that accept a `DistanceMetric`, we can switch between distances at run-time:

```cpp
std::shared_ptr<knncolle::DistanceMetric<double, double> > distptr;
if (distance == "euclidean") {
    distptr.reset(new knncolle::EuclideanDistance<double, double>);
} else if (distance == "manhattan") {
    distptr.reset(new knncolle::ManhattanDistance<double, double>);
} else {
    // do something else.
}

knncolle::VptreeBuilder<int, double, double> vp_builder(std::move(distptr));
```

We can even switch between input matrix representations at run-time, as long as they follow the `Matrix` interface.
This allows the various `Builder` classes to accept input data in other formats (e.g., sparse, file-backed).
For example, **knncolle** implements the `L2NormalizedMatrix` subclass to apply on-the-fly L2 normalization of each observation's vector of coordinates.
This is used inside the `L2NormalizedBuilder` class to transform an existing neighbor search method from Euclidean to cosine distances.

```cpp
auto builder = std::make_shared<knncolle::VptreeBuilder<int, double, double> >(
    std::make_shared<knncolle::EuclideanDistance<double, double> >()
);

auto l2builder = std::make_shared<knncolle::L2NormalizedBuilder<
    /* observation index */ int,
    /* data type */ double,
    /* distance type */ double,
    /* normalized type */ double
> >(std::move(builder));

// Any Matrix 'mat' is automatically wrapped in a L2NormalizedMatrix
// before being passed to 'builder->build_unique'.
auto l2index = l2builder->build_unique(mat);
```

Check out the [reference documentation](https://knncolle.github.io/knncolle/) for more details on these interfaces.

## Modifying template parameters

Each interface has a few template parameters to define its types.
In general, we recommend using `int`s for the observation indices and `double`s for the data and distances.
If precision is not a concern, we can achieve greater speed by swapping `double`s with `float`s.
We may also need to swap `int` with `size_t` for larger datasets, e.g., more than 2 billion observations.

Advanced users can set up the templates to bypass virtual dispatch at the cost of more compile-time complexity.
For example, we could parametrize the `VptreeBuilder` so that it is hard-coded to use Euclidean distances and to only accept column-major in-memory matrices.
This gives the compiler an opportunity to devirtualize the relevant method calls for a potential performance improvement.

```cpp
typedef knncolle::VptreeBuilder<
    int,
    double,
    double,
    kncolle::SimpleMatrix<int, double>,
    knncolle::EuclideanDistance<double, double>
> VptreeEuclideanSimple;
```

## Saving indices to file

Once a `Prebuilt` search index is constructed, we can save it to file.
We can then load it back into memory on its next use, thus avoiding the need to rebuild the same index.

```cpp
vp_index->save("foo/bar_");

// This should be called with the same template parameters as the Prebuilt
// object that was saved, i.e., Index_, Data_ and Distance_.
auto reloaded = knncolle::load_prebuilt_shared<int, double, double>("foo/bar_");
```

For L2-normalized matrices, the situation is a little trickier as we need to consider the data type after normalization.
Here, we specify the supported set of normalized types for our loading function:

```cpp
auto& reg = knncolle::load_prebuilt_registry();
reg[knncolle::l2normalized_save_name] = [](const std::string& prefix) -> knncolle::Prebuilt<int, double, double>* {
    auto config = knncolle::load_l2normalized_prebuilt_types();

    // Possibly could support float, if we expected to get single-precision saved indices.
    if (config.normalized != knncolle::get_numeric_type<double>()) {
        throw std::runtime_error("don't know how to handle non-double normalized types");
    }

    return knncolle::load_l2normalized_prebuilt<int, double, double, double>(prefix);
};
```

For ease of implementation, the files generated by `Prebuilt::save()` are not guaranteed to be portable across architectures, compilers, or even versions of **knncolle**.
An index saved by an application is only intended to be read by the same application on the same machine.

## Building projects with **knncolle**

### CMake with `FetchContent`

If you're using CMake, you just need to add something like this to your `CMakeLists.txt`:

```cmake
include(FetchContent)

FetchContent_Declare(
  knncolle
  GIT_REPOSITORY https://github.com/knncolle/knncolle
  GIT_TAG master # replace with a pinned release
)

FetchContent_MakeAvailable(knncolle)
```

Then you can link to **knncolle** to make the headers available during compilation:

```cmake
# For executables:
target_link_libraries(myexe knncolle::knncolle)

# For libaries
target_link_libraries(mylib INTERFACE knncolle::knncolle)
```

By default, this will use `FetchContent` to fetch all external dependencies.
Applications are advised to pin the versions of each dependency for stability - see [`extern/CMakeLists.txt`](extern/CMakeLists.txt) to find suggested versions.
If you want to install dependencies manually, set `-DKNNCOLLE_FETCH_EXTERN=OFF` in the Cmake configuration.

### CMake with `find_package()`

If **knncolle** is already installed on the system, it can be discovered via:

```cmake
find_package(knncolle_knncolle CONFIG REQUIRED)
target_link_libraries(mylib INTERFACE knncolle::knncolle)
```

To install the library, use:

```sh
mkdir build && cd build
cmake .. -DKNNCOLLE_TESTS=OFF
cmake --build . --target install
```

Again, this will automatically acquire all its dependencies, see recommendations above.

### Manual

If you're not using CMake, the simple approach is to just copy the files in `include/` - either directly or with Git submodules - 
and include their path during compilation with, e.g., GCC's `-I`.
This will also require the external dependencies listed in [`extern/CMakeLists.txt`](extern/CMakeLists.txt). 

## References

Hanov S (2011).
VP trees: A data structure for finding stuff fast.
http://stevehanov.ca/blog/index.php?id=130

Yianilos PN (1993).
Data structures and algorithms for nearest neighbor search in general metric spaces.
_Proceedings of the Fourth Annual ACM-SIAM Symposium on Discrete Algorithms_, 311-321.
