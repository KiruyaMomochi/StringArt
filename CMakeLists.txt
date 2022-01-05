# It's possible to lower minimum version required by changing some of this
# file, see README.md for more information.
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(StringArt
    VERSION 0.1.0 
    DESCRIPTION "String Art"
    LANGUAGES CUDA CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(CTest)
enable_testing()

file(DOWNLOAD https://github.com/cpm-cmake/CPM.cmake/releases/latest/download/get_cpm.cmake
    ${CMAKE_BINARY_DIR}/CPM.cmake)
include (${CMAKE_BINARY_DIR}/CPM.cmake)

CPMAddPackage("gh:fmtlib/fmt@8.0.1#8.0.1")
CPMAddPackage("gh:eyalroz/cuda-api-wrappers@0.4.4")
CPMAddPackage("gh:eyalroz/cuda-api-wrappers@0.4.4")

add_subdirectory(src/stb/)
add_subdirectory(src/StringArt/)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
