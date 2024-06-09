include(FetchContent)

FetchContent_Declare(
  kmeans
  GIT_REPOSITORY https://github.com/LTLA/CppKmeans
  GIT_TAG master
)

FetchContent_MakeAvailable(kmeans)
