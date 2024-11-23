target_include_directories(fgprs PUBLIC deeplab)
include_directories(deeplab)
add_subdirectory(deeplab)

target_sources(fgprs PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/resnet.cpp)