#ifndef KNNCOLLE_HPP
#define KNNCOLLE_HPP

#include "Builder.hpp"
#include "Prebuilt.hpp"
#include "MockMatrix.hpp"
#include "distances.hpp"

#include "Bruteforce.hpp"
#include "Vptree.hpp"
#include "Kmknn.hpp"
#include "find_nearest_neighbors.hpp"

/**
 * @file knncolle.hpp
 *
 * @brief Umbrella header to include all algorithms.
 *
 * Developers can avoid the inclusion of unnecessary dependencies by setting:
 *
 * - `KNNCOLLE_NO_KMKNN`, to avoid including the `Kmknn.hpp` header (which requires the **kmeans** library).
 * - `KNNCOLLE_NO_ANNOY`, to avoid including the `Annoy.hpp` header (which requires the **Annoy** library).
 * - `KNNCOLLE_NO_HNSW`, to avoid including the `Hnsw.hpp` header (which requires the **Hnsw** library).
 */

#endif

