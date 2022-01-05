add_executable(StringArtSequential
               main.cpp)

target_link_libraries(StringArtSequential
                      StringArt
                      fmt::fmt
                      stb)
