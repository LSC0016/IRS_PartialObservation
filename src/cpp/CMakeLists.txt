# src/cpp/CMakeLists.txt
cmake_minimum_required(VERSION 3.10)
project(DataGenerator)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 查找 OpenMP
find_package(OpenMP REQUIRED)

# 查找 PyBind11
find_package(pybind11 REQUIRED)

# 添加 PyBind11 模块
pybind11_add_module(data_generator_cpp data_generator_bindings.cpp data_generator.cpp)

# 链接 OpenMP
target_link_libraries(data_generator_cpp PRIVATE OpenMP::OpenMP_CXX)
