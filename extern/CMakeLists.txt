include(FetchContent)

FetchContent_Declare(
  kmeans
  GIT_REPOSITORY https://github.com/LTLA/CppKmeans
  GIT_TAG master # ^3.0.0
)

FetchContent_MakeAvailable(kmeans)
