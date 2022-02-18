cmake_minimum_required (VERSION 3.10)

project (SCI)
set(NAME "SCI")

option(SCI_BUILD_TESTS "Build tests" OFF)
message(STATUS "Option: SCI_BUILD_TESTS = ${SCI_BUILD_TESTS}")

option(SCI_BUILD_NETWORKS "Build networks" OFF)
message(STATUS "Option: SCI_BUILD_NETWORKS = ${SCI_BUILD_NETWORKS}")

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_subdirectory(src)

if (SCI_BUILD_TESTS)
    add_subdirectory(tests)
endif()

if (SCI_BUILD_NETWORKS)
    add_subdirectory(networks)
endif()
