cmake_minimum_required(VERSION 3.4)
project(ToolKit)
set(CMAKE_CXX_STANDAND 11)
set(NCNN_INSTALL_SDK OFF CACHE BOOL "Install ncnn library and headers")
add_subdirectory(src)
add_subdirectory(ncnn)
add_subdirectory(MNN)
add_subdirectory(Tengine)
