add_executable(StringArtCuda
               main.cu)

target_link_libraries(StringArtCuda
                      StringArt
                      fmt::fmt
                      stb
                      spdlog::spdlog)
