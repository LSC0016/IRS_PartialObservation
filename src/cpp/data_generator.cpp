// src/cpp/data_generator.cpp
#include "data_generator.h"
#include <cmath>
#include <random>
#include <iostream>
#include <omp.h> // OpenMP header
#include <fstream>
#include <sstream>
#include <numeric>   // For std::accumulate
#include <algorithm> // For std::transform

namespace DataGenerator
{

    // Function to compute the average signal power
    double avg_signal_pw(
        const std::map<std::string, std::vector<int>> &assign_dict,
        double beta,
        double alpha,
        int nPU,
        int nSU,
        int nch,
        double DistAmp,
        const std::map<int, std::vector<double>> &dist_dict)
    {
        double total_pw = 0.0;

// Use OpenMP parallelization with reduction
#pragma omp parallel for reduction(+ : total_pw)
        for (int SU = 0; SU < nSU; ++SU)
        {
            for (int PU = 0; PU < nPU; ++PU)
            {
                std::string pu_key = "PU" + std::to_string(PU);
                double distance = DistAmp * dist_dict.at(PU)[SU];
                double gain = 1.0 / (beta * std::pow(distance, alpha));
                int num_bands = assign_dict.at(pu_key).size();
                total_pw += num_bands * gain;
            }
        }
        double avg_pw = total_pw / (nSU * nch);
        std::cout << "Average power per frequency point: " << avg_pw << std::endl;
        return avg_pw;
    }

    // Function to load PSD library data
    PSDLibrary load_PSD_library(const std::string &directory)
    {
        PSDLibrary psd_lib;
        psd_lib.description = "Loaded from CSV files";

        for (int PU = 1; PU <= 6; ++PU)
        {
            std::vector<std::vector<double>> psd_samples;

            std::ifstream file(directory + "/PSD_PU" + std::to_string(PU) + ".csv");
            if (!file.is_open())
            {
                std::cerr << "Failed to open PSD file for PU" << PU << std::endl;
                continue;
            }

            std::string line;
            while (std::getline(file, line))
            {
                std::vector<double> sample;
                std::stringstream ss(line);
                std::string value;
                while (std::getline(ss, value, ','))
                {
                    sample.push_back(std::stod(value));
                }
                psd_samples.push_back(sample);
            }

            psd_lib.data[PU] = psd_samples;
        }

        return psd_lib;
    }

    // Data generator function implementation
    std::tuple<
        std::vector<std::vector<std::vector<std::vector<double>>>>,
        std::vector<std::vector<int>>>
    generate_data(
        double DistAmp,
        const std::vector<std::vector<int>> &class_dir,
        const std::vector<int> &dbsize_list,
        int nch,
        int nw,
        const std::map<std::string, std::vector<int>> &assign_dict,
        double SNR,
        const std::map<int, std::vector<double>> &dist_dict,
        const PSDLibrary &PSD_lib,
        double alpha,
        double beta)
    {
        int nPU = assign_dict.size() - 1; // Exclude 'description' if present
        int nSU = class_dir.size();
        int num_classes = dbsize_list.size();
        std::vector<std::vector<std::vector<std::vector<double>>>> db;
        std::vector<std::vector<int>> label_list;

        // Compute average signal power
        double avg_pw = avg_signal_pw(assign_dict, beta, alpha, nPU, nSU, nch, DistAmp, dist_dict);
        // Compute noise power based on SNR
        double noi_pw = avg_pw * std::pow(10, -SNR / 10.0);

        // Random number generators
        std::random_device rd;
        std::mt19937 gen(rd());

// OpenMP parallelization
#pragma omp parallel for schedule(dynamic)
        for (int cls = 0; cls < num_classes; ++cls)
        {
            int db_size = dbsize_list[cls];
            for (int n = 0; n < db_size; ++n)
            {
                std::vector<std::vector<std::vector<double>>> inp(nSU);
                std::vector<int> label(nch, 0);
                std::vector<int> PSDidx(nPU);

                // Each thread should have its own random number generator
                std::mt19937 thread_gen(rd() ^ (omp_get_thread_num() + 1));

                for (int PU = 0; PU < nPU; ++PU)
                {
                    std::uniform_int_distribution<> PSD_idx_dist(0, PSD_lib.data.at(PU + 1).size() - 1);
                    PSDidx[PU] = PSD_idx_dist(thread_gen);
                }

                for (int SU = 0; SU < nSU; ++SU)
                {
                    std::vector<std::vector<double>> a(1, std::vector<double>(nw * nch, 0.0));

                    // Add noise
                    std::normal_distribution<> noise_dist(0, noi_pw);
                    for (auto &val : a[0])
                    {
                        val += noise_dist(thread_gen);
                    }

                    for (int PU = 0; PU < nPU; ++PU)
                    {
                        // Check if PU is active in this class; this needs to be defined based on your logic
                        bool pu_active = true; // Placeholder; you should define the actual condition

                        if (pu_active)
                        {
                            double adjusted_dist = DistAmp * dist_dict.at(PU)[SU];
                            double ch_gain = 1.0 / (beta * std::pow(adjusted_dist, alpha));

                            // Apply path loss and shadow fading
                            std::normal_distribution<> shadow_fading(0, 0.365);
                            ch_gain *= std::pow(10, -0.365 * shadow_fading(thread_gen));

                            for (int ch : assign_dict.at("PU" + std::to_string(PU)))
                            {
                                label[ch] = 1;

                                // Get PSD signal
                                const auto &rcv_sig = PSD_lib.data.at(PU + 1)[PSDidx[PU]];

                                // Scale the signal
                                double pw = std::accumulate(rcv_sig.begin(), rcv_sig.end(), 0.0) / rcv_sig.size();
                                std::vector<double> scaled_sig(rcv_sig.size());
                                std::transform(rcv_sig.begin(), rcv_sig.end(), scaled_sig.begin(),
                                               [ch_gain, pw](double val)
                                               {
                                                   return val / pw * ch_gain;
                                               });

                                // Add to the signal
                                for (int idx = 0; idx < nw; ++idx)
                                {
                                    if (ch * nw + idx < a[0].size())
                                    {
                                        a[0][ch * nw + idx] += scaled_sig[idx];
                                    }
                                }
                            }
                        }
                    }
                    inp[SU] = a;
                }

// Protect shared resources
#pragma omp critical
                {
                    db.push_back(inp);
                    label_list.push_back(label);
                }
            }
        }

        return std::make_tuple(db, label_list);
    }

} // namespace DataGenerator
