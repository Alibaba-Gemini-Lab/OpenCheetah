add_library(SCI-BuildingBlocks
    value-extension.cpp
    aux-protocols.cpp
    truncation.cpp)

target_link_libraries(SCI-BuildingBlocks
    PUBLIC SCI-OTPrimitive SCI-Millionaire
)

# Cheetah related
add_library(SCI-Cheetah-BuildingBlocks value-extension.cpp aux-protocols.cpp truncation.cpp)
target_link_libraries(SCI-Cheetah-BuildingBlocks PUBLIC SCI-OTPrimitive SCI-Millionaire)
target_compile_definitions(SCI-Cheetah-BuildingBlocks PUBLIC USE_CHEETAH=1)
