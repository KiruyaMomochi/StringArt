set(STB_FILES stb_image stb_image_resize stb_image_write)

foreach(file ${STB_FILES})
    file(DOWNLOAD "https://raw.githubusercontent.com/nothings/stb/master/${file}.h"
        "${CMAKE_CURRENT_BINARY_DIR}/include/${file}.h"
    )
endforeach()

add_library(stb INTERFACE)
add_library(StringArt::stb ALIAS stb)
target_include_directories(stb INTERFACE "${CMAKE_CURRENT_BINARY_DIR}/include/")
set_target_properties(stb PROPERTIES LINKER_LANGUAGE CXX)
