# Collection of KNN algorithms

![Unit tests](https://github.com/LTLA/knncolle/actions/workflows/run-tests.yaml/badge.svg)
![Documentation](https://github.com/LTLA/knncolle/actions/workflows/doxygenate.yaml/badge.svg)

## Overview

This is a header-only C++ library that collects a variety of different k-nearest neighbor algorithms under a consistent interface.
As one might expect, the name is not an accident.

<p float="left">
  <img src="https://i.makeagif.com/media/2-26-2015/JDQzgr.gif" width="32%" />
  <img src="https://thumbs.gfycat.com/SneakyPracticalIndianringneckparakeet-max-1mb.gif" width="32%" />
  <img src="https://media.tenor.com/images/2b3d5c70f6f4919320480f13427d881c/tenor.gif" width="32%" />
</p>

The aim of **knncolle** is to enable downstream libraries to easily switch between different methods with a single runtime flag,
or by just swapping out the relevant constructors at compile time.
Currently, we support the following methods:

- [Vantage point tree](http://stevehanov.ca/blog/?id=130)
- [Annoy](https://github.com/spotify/annoy/)
- [HNSW](https://github.com/nmslib/hnswlib/)
- Brute force search

## Quick start

Given a matrix with dimensions in the rows and observations in the columns, we can do:

```cpp
#include "knncolle/knncolle.hpp"

/* ... boilerplate... */

knncolle::Dispatch<> dispatcher;
knncolle::DispatchAlgorithm algo_choice = knncolle::VPTREE; // or ANNOY, or HNSW...
auto searcher = dispatcher.build(ndim, nobs, matrix.data(), algo_choice);

auto results1 = searcher->find_nearest_neighbors(0, 10); // 10 nearest neighbors of the first element.
auto results2 = searcher->find_nearest_neighbors(query, 10); // 10 nearest neighbors of a query vector.
```

The `find_nearest_neighbors()` call will return a vector of (index, distance) pairs,
containing the requested number of neighbors in order of increasing distance from the query point.
(In cases where the requested number of neighbors is greater than the actual number of neighbors, the latter is returned.)
For some algorithms, we can modify the parameters of the search by setting the relevant members of the `Dispatch` class:

```cpp
dispatcher.Annoy.ntrees = 100;
auto searcher_annoy = dispatcher.build(ndim, nobs, matrix.data(), knncolle::ANNOY);
```

If the desired algorithm is known at compile time, we can be more specific:

```cpp
knncolle::AnnoyEuclidean<> searcher(ndim, nobs, matrix.data(), /* ntrees = */ 100); 
```

Check out the [reference documentation](https://ltla.github.io/knncolle/) for more details.

## Building projects with **knncolle**

If you're using CMake, you just need to add something like this to your `CMakeLists.txt`:

```
include(FetchContent)

FetchContent_Declare(
  knncolle
  GIT_REPOSITORY https://github.com/LTLA/knncolle
  GIT_TAG master # or any version of interest
)

FetchContent_MakeAvailable(knncolle)
```

Then you can link to **knncolle** to make the headers available during compilation:

```
# For executables:
target_link_libraries(myexe knncolle)

# For libaries
target_link_libraries(mylib INTERFACE knncolle)
```

If you're not using CMake, the simple approach is to just copy the files - either directly or with Git submodules - and include their path during compilation with, e.g., GCC's `-I`.
Note that this will require the manual inclusion of all dependencies, namely the [Annoy](https://github.com/spotify/annoy) and [HNSW](https://github.com/nmslib/hsnwlib) libraries.

# TODO:

- Add KMKNN as another exact method.
