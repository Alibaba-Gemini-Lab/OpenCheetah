add_library(Cheetah-Linear cheetah-api.cpp)
target_link_libraries(Cheetah-Linear
  PUBLIC
  SCI-Cheetah-BuildingBlocks
  Eigen3::Eigen
  SEAL::seal)
target_compile_definitions(Cheetah-Linear PUBLIC USE_CHEETAH=1)
