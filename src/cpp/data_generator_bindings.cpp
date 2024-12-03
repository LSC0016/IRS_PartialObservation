// src/cpp/data_generator_bindings.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "data_generator.h"

namespace py = pybind11;

PYBIND11_MODULE(data_generator_cpp, m)
{
    m.doc() = "C++ Data Generator Module";

    py::class_<PSDLibrary>(m, "PSDLibrary")
        .def(py::init<>())
        .def_readwrite("description", &PSDLibrary::description)
        .def_readwrite("data", &PSDLibrary::data);

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

    m.def("load_PSD_library", &DataGenerator::load_PSD_library, "Load PSD library from directory",
          py::arg("directory"));
}
