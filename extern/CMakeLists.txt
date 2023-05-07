include(FetchContent)

FetchContent_Declare(
  Annoy 
  GIT_REPOSITORY https://github.com/spotify/Annoy
  GIT_TAG v1.17.2
)

FetchContent_MakeAvailable(Annoy)

FetchContent_Declare(
  hnswlib
  GIT_REPOSITORY https://github.com/nmslib/hnswlib
  GIT_TAG dccd4f98acb9da404b7439606a97c4e3077a8d44 # v0.7.0 + CMake targets from develop
)

FetchContent_MakeAvailable(hnswlib)

FetchContent_Declare(
  kmeans
  GIT_REPOSITORY https://github.com/LTLA/CppKmeans
  GIT_TAG 4c5aca44bffd8ed7d7243b2451105b572028e9d4
)

FetchContent_MakeAvailable(kmeans)
