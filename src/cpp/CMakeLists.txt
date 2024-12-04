cmake_minimum_required(VERSION 3.10)
project(DataGenerator)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 查找 OpenMP
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_CXX_LINK_FLAGS "${CMAKE_CXX_LINK_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# 查找 PyBind11
find_package(pybind11 REQUIRED)

# 查找 CUDA
find_package(CUDA REQUIRED)

# 添加头文件目录
include_directories(${pybind11_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)

# 添加源文件
set(OPENMP_SOURCE_FILES data_generator_openmp.cpp data_generator_bindings.cpp)
set(CUDA_SOURCE_FILES data_generator_openmp.cpp data_generator_cuda.cu data_generator_bindings.cpp)

# 添加 PyBind11 模块 - OpenMP 版本
pybind11_add_module(data_generator_openmp_cpp MODULE ${OPENMP_SOURCE_FILES})
target_link_libraries(data_generator_openmp_cpp PRIVATE OpenMP::OpenMP_CXX)

# 添加 PyBind11 模块 - CUDA 版本
# 编译 CUDA 文件为静态库并链接
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3 --use_fast_math -std=c++14")
cuda_add_library(data_generator_cuda STATIC data_generator_cuda.cu)
pybind11_add_module(data_generator_cuda_cpp MODULE ${CUDA_SOURCE_FILES})
target_link_libraries(data_generator_cuda_cpp PRIVATE data_generator_cuda)

# 设置调试和优化选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O3")




