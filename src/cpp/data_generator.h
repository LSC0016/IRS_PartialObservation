// src/cpp/data_generator.h
#ifndef DATA_GENERATOR_H
#define DATA_GENERATOR_H

#include <vector>
#include <map>
#include <string>
#include <tuple>

// 定义PSD库的数据结构
struct PSDLibrary {
    std::string description;
    std::map<int, std::vector<std::vector<double>>> data; // PU索引到PSD数据的映射
};

// 数据生成器命名空间
namespace DataGenerator {

    // 数据生成器函数，返回生成的数据和标签
    std::tuple<
        std::vector<std::vector<std::vector<std::vector<double>>>>, // 数据列表
        std::vector<std::vector<int>>                               // 标签列表
    >
    generate_data(
        double DistAmp,
        const std::vector<std::vector<int>>& class_dir,
        const std::vector<int>& dbsize_list,
        int nch,
        int nw,
        const std::map<std::string, std::vector<int>>& assign_dict,
        double SNR,
        const std::map<int, std::vector<double>>& dist_dict,
        const PSDLibrary& PSD_lib,
        double alpha = 3.71,
        double beta = std::pow(10, 3.154)
    );

    // 加载PSD库数据的函数
    PSDLibrary load_PSD_library(const std::string& directory);

}

#endif // DATA_GENERATOR_H
