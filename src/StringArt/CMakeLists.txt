add_executable(StringArt
               main.cpp)

target_link_libraries(StringArt
                      PUBLIC 
                      fmt::fmt
                      nvtx
                      stb)

# target_include_directories(StringArt PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
