cmake_minimum_required(VERSION 3.10)
project(det)

set(CMAKE_CXX_STANDARD 11)


# cuda
find_package(CUDA REQUIRED)
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)


# tensorrt
#include_directories(/usr/include/x86_64-linux-gnu)
#link_directories(/usr/lib/x86_64-linux-gnu/)
include_directories(/usr/local/TensorRT-7.2.2.3/include/)
link_directories(/usr/local/TensorRT-7.2.2.3/lib/)


add_executable(det main.cpp)

target_link_libraries(det nvinfer cudart)
add_definitions(-O2 -pthread)
