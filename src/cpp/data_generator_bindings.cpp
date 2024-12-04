#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "data_generator.h"

namespace py = pybind11;

// OpenMP 版本的模块
PYBIND11_MODULE(data_generator_openmp_cpp, m)
{
    m.doc() = "C++ Data Generator Module using OpenMP";

    // PSDLibrary class
    py::class_<PSDLibrary>(m, "PSDLibrary")
        .def(py::init<>())
        .def_readwrite("description", &PSDLibrary::description)
        .def_readwrite("data", &PSDLibrary::data);

    // Generate data function
    m.def("generate_data", &DataGenerator::generate_data, "Generate data with given parameters",
          py::arg("DistAmp"),
          py::arg("class_dir"),
          py::arg("dbsize_list"),
          py::arg("nch"),
          py::arg("nw"),
          py::arg("assign_dict"),
          py::arg("SNR"),
          py::arg("dist_dict"),
          py::arg("PSD_lib"),
          py::arg("alpha") = 3.71,
          py::arg("beta") = std::pow(10, 3.154));

    // Load PSD library function
    m.def("load_PSD_library", &DataGenerator::load_PSD_library, "Load PSD library from directory",
          py::arg("directory"));
}

// CUDA 版本的模块
PYBIND11_MODULE(data_generator_cuda_cpp, m)
{
    m.doc() = "C++ Data Generator Module using CUDA";

    // PSDLibrary class
    py::class_<PSDLibrary>(m, "PSDLibrary")
        .def(py::init<>())
        .def_readwrite("description", &PSDLibrary::description)
        .def_readwrite("data", &PSDLibrary::data);

    // CUDA 加速数据生成函数
    m.def("generate_data", &DataGenerator::generate_data, "Generate data using CUDA",
          py::arg("DistAmp"),
          py::arg("class_dir"),
          py::arg("dbsize_list"),
          py::arg("nch"),
          py::arg("nw"),
          py::arg("assign_dict"),
          py::arg("SNR"),
          py::arg("dist_dict"),
          py::arg("PSD_lib"),
          py::arg("alpha") = 3.71,
          py::arg("beta") = std::pow(10, 3.154));

    // Load PSD library function
    m.def("load_PSD_library", &DataGenerator::load_PSD_library, "Load PSD library from directory",
          py::arg("directory"));
}
