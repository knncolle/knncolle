include(FetchContent)

FetchContent_Declare(
  Annoy 
  GIT_REPOSITORY https://github.com/LTLA/Annoy
  GIT_TAG cmake 
)

FetchContent_MakeAvailable(Annoy)

FetchContent_Declare(
  hnswlib
  GIT_REPOSITORY https://github.com/LTLA/hnswlib
  GIT_TAG master
)

FetchContent_MakeAvailable(hnswlib)
