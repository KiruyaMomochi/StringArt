add_executable(StringArtCuda
               main.cu)

target_link_libraries(StringArtCuda
                      StringArt::StringArt
                      fmt::fmt
                      StringArt::stb
                      spdlog::spdlog
                      cxxopts::cxxopts)
