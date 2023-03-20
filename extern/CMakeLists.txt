include(FetchContent)

FetchContent_Declare(
  Annoy 
  GIT_REPOSITORY https://github.com/spotify/Annoy
  GIT_TAG 0457742d12cf7a263f7c709ef1d470e5f08f791d
)

FetchContent_MakeAvailable(Annoy)

FetchContent_Declare(
  hnswlib
  GIT_REPOSITORY https://github.com/nmslib/hnswlib
  GIT_TAG v0.7.0
)

# Avoid FetchContent_MakeAvailable() to not use add_subdirectory, because hnswlib CMake does not have install targets.
# To make FetchContent compatible with install it is required to use the find module instead.
FetchContent_GetProperties(hnswlib)
if(NOT hnswlib_POPULATED)
  FetchContent_Populate(hnswlib)
endif()
set(hnswlib_ROOT ${hnswlib_SOURCE_DIR})
find_package(hnswlib REQUIRED GLOBAL)

FetchContent_Declare(
  kmeans
  GIT_REPOSITORY https://github.com/LTLA/CppKmeans
  GIT_TAG 4c5aca44bffd8ed7d7243b2451105b572028e9d4
)

FetchContent_MakeAvailable(kmeans)
