add_library(StringArt StringArt.cpp)

target_link_libraries(StringArt
                      fmt::fmt
                      nvtx
                      stb)

target_include_directories(StringArt PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
