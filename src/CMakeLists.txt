cmake_minimum_required(VERSION 3.4)

set(CMAKE_CXX_STANDAND 11)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../ncnn/src)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/../ncnn/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../MNN/source)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/../MNN/source)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../MNN/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../MNN/include/MNN)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../MNN/tools/cpp)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../MNN/schema/current)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../MNN/3rd_party/flatbuffers/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../Tengine/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../Tengine/core/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../Tengine/benchmark/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../Tengine/executor/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../Tengine/operator/include)
file(GLOB MODEL ${CMAKE_CURRENT_LIST_DIR}/model.h
		${CMAKE_CURRENT_LIST_DIR}/model.cpp
		${CMAKE_CURRENT_LIST_DIR}/main.cpp
		${CMAKE_CURRENT_SOURCE_DIR}/../MNN/tools/cpp/revertMNNModel.cpp)
set(MODEL_CODE ${MODEL})
add_executable(benchmark ${MODEL_CODE})
target_link_libraries(benchmark ncnn MNN tengine)
install(TARGETS benchmark DESTINATION lib)
