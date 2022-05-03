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
  GIT_TAG d6c8e3a3896129a7f642bc23cdce4536e8b45a65
)

FetchContent_MakeAvailable(hnswlib)

FetchContent_Declare(
  kmeans
  GIT_REPOSITORY https://github.com/LTLA/CppKmeans
  GIT_TAG 4397a8d576cf0b657fd9012c049e05727c45796d
)

FetchContent_MakeAvailable(kmeans)
