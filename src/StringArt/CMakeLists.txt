add_library(StringArt StringArt.cpp)

target_link_libraries(StringArt
                      fmt::fmt
                      nvtx
                      stb
                      spdlog::spdlog)

target_include_directories(StringArt PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_options(StringArt PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>) 
