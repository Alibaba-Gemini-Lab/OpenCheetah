add_library(SCI-LinearHE
    conv-field.cpp
    fc-field.cpp
    elemwise-prod-field.cpp
    utils-HE.cpp)

target_link_libraries(SCI-LinearHE
    PUBLIC
    SCI-utils
    SEAL::seal
    Eigen3::Eigen)
