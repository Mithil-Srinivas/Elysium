cmake_minimum_required(VERSION 3.29)
project(elysium C)

set(CMAKE_C_STANDARD 11)

include_directories("${CMAKE_SOURCE_DIR}/ten/include")
include_directories("${CMAKE_SOURCE_DIR}/onnx/include")
include_directories("${CMAKE_SOURCE_DIR}/stb")

link_directories("${CMAKE_SOURCE_DIR}/ten/lib")
link_directories("${CMAKE_SOURCE_DIR}/onnx/lib")

add_executable(elysium newdet.c rec.c)

target_link_libraries(elysium "${CMAKE_SOURCE_DIR}/onnx/lib/onnxruntime.lib")