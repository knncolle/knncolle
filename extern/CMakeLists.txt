include(FetchContent)

FetchContent_Declare(
  Annoy 
  GIT_REPOSITORY https://github.com/spotify/Annoy
  GIT_TAG 02e6de1c40dfe78a984e5a6f0611d0c6513ff875
)

FetchContent_MakeAvailable(Annoy)

FetchContent_Declare(
  hnswlib
  GIT_REPOSITORY https://github.com/LTLA/hnswlib
  GIT_TAG master
)

FetchContent_MakeAvailable(hnswlib)

FetchContent_Declare(
  kmeans
  GIT_REPOSITORY https://github.com/LTLA/CppKmeans
  GIT_TAG master
)

FetchContent_MakeAvailable(kmeans)
