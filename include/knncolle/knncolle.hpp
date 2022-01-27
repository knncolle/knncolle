#ifndef KNNCOLLE_HPP
#define KNNCOLLE_HPP

#include "BruteForce/BruteForce.hpp"
#include "VpTree/VpTree.hpp"

#ifndef KNNCOLLE_NO_KMKNN
#include "Kmknn/Kmknn.hpp"
#endif

#ifndef KNNCOLLE_NO_ANNOY
#include "Annoy/Annoy.hpp"
#endif

#ifndef KNNCOLLE_NO_HNSW
#include "Hnsw/Hnsw.hpp"
#endif

#include "utils/find_nearest_neighbors.hpp"

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

