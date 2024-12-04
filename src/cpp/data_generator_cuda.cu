// src/cpp/data_generator_cuda.cu

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <map>
#include <vector>
#include <iostream>
#include "data_generator.h"

// 自定义的 atomicAdd 函数，用于 double 类型
__device__ double atomicAddDouble(double *address, double value)
{
    unsigned long long int *address_as_ull =
        (unsigned long long int *)address;
    unsigned long long int old = *address_as_ull, assumed;
    do
    {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(value + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

// CUDA 核函数用于计算平均信号功率
__global__ void avg_signal_pw_kernel(double *d_total_pw, const double *d_distances, int nPU, int nSU, double beta, double alpha, double DistAmp, int nBandsPerPU)
{
    int suIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (suIdx < nSU)
    {
        for (int PU = 0; PU < nPU; ++PU)
        {
            double distance = DistAmp * d_distances[PU * nSU + suIdx];
            double gain = 1.0 / (beta * pow(distance, alpha));
            double power = nBandsPerPU * gain;

            // 使用自定义的 atomicAddDouble 函数避免数据竞争
            atomicAddDouble(d_total_pw, power);
        }
    }
}

// 主机函数，用于调用 CUDA 核函数计算平均信号功率
double avg_signal_pw_cuda(
    const std::map<int, std::vector<double>> &dist_dict,
    double beta,
    double alpha,
    int nPU,
    int nSU,
    int nch,
    double DistAmp,
    int nBandsPerPU)
{
    // 分配和初始化设备内存
    double *d_distances;
    cudaMalloc((void **)&d_distances, nPU * nSU * sizeof(double));

    double *distances = new double[nPU * nSU];
    for (int PU = 0; PU < nPU; ++PU)
    {
        for (int SU = 0; SU < nSU; ++SU)
        {
            distances[PU * nSU + SU] = dist_dict.at(PU)[SU];
        }
    }
    cudaMemcpy(d_distances, distances, nPU * nSU * sizeof(double), cudaMemcpyHostToDevice);

    double total_pw = 0.0;
    double *d_total_pw;
    cudaMalloc((void **)&d_total_pw, sizeof(double));
    cudaMemcpy(d_total_pw, &total_pw, sizeof(double), cudaMemcpyHostToDevice);

    // 启动 CUDA 核函数
    int threadsPerBlock = 256;
    int blocksPerGrid = (nSU + threadsPerBlock - 1) / threadsPerBlock;
    avg_signal_pw_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_total_pw, d_distances, nPU, nSU, beta, alpha, DistAmp, nBandsPerPU);

    // 同步并获取结果
    cudaMemcpy(&total_pw, d_total_pw, sizeof(double), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    double avg_pw = total_pw / (nSU * nch);

    // 清理设备内存
    cudaFree(d_distances);
    cudaFree(d_total_pw);
    delete[] distances;

    return avg_pw;
}

// CUDA 核函数用于生成数据
__global__ void generate_data_kernel(double *d_data, double *d_noises, const double *d_distances, int nPU, int nSU, int nch, int nw, double beta, double alpha, double DistAmp, double noi_pw)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalElements = nSU * nch * nw;
    if (idx < totalElements)
    {
        int suIdx = idx / (nch * nw);
        int chIdx = (idx / nw) % nch;
        int sampleIdx = idx % nw;

        // 加入噪声
        d_data[idx] = d_noises[idx];

        for (int PU = 0; PU < nPU; ++PU)
        {
            double distance = DistAmp * d_distances[PU * nSU + suIdx];
            double ch_gain = 1.0 / (beta * pow(distance, alpha));

            // 对信道增益进行阴影衰落处理（此处可加入随机衰落，但为简单起见省略）
            // ch_gain *= shadow_fading_factor;

            // 将信号加权并添加到数据中
            d_data[idx] += ch_gain; // 简化版，仅为演示
        }
    }
}

// 主机函数，用于调用 CUDA 核函数生成数据
std::vector<std::vector<std::vector<double>>> generate_data_cuda(
    const std::map<int, std::vector<double>> &dist_dict,
    double beta,
    double alpha,
    int nPU,
    int nSU,
    int nch,
    int nw,
    double DistAmp,
    double noi_pw)
{

    // 分配和初始化设备内存
    double *d_distances;
    cudaMalloc((void **)&d_distances, nPU * nSU * sizeof(double));

    double *distances = new double[nPU * nSU];
    for (int PU = 0; PU < nPU; ++PU)
    {
        for (int SU = 0; SU < nSU; ++SU)
        {
            distances[PU * nSU + SU] = dist_dict.at(PU)[SU];
        }
    }
    cudaMemcpy(d_distances, distances, nPU * nSU * sizeof(double), cudaMemcpyHostToDevice);

    double *d_data;
    cudaMalloc((void **)&d_data, nSU * nch * nw * sizeof(double));

    double *d_noises;
    cudaMalloc((void **)&d_noises, nSU * nch * nw * sizeof(double));

    // 随机初始化噪声数据
    double *noises = new double[nSU * nch * nw];
    for (int i = 0; i < nSU * nch * nw; ++i)
    {
        noises[i] = noi_pw * ((double)rand() / RAND_MAX);
    }
    cudaMemcpy(d_noises, noises, nSU * nch * nw * sizeof(double), cudaMemcpyHostToDevice);

    // 启动 CUDA 核函数生成数据
    int threadsPerBlock = 256;
    int blocksPerGrid = (nSU * nch * nw + threadsPerBlock - 1) / threadsPerBlock;
    generate_data_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, d_noises, d_distances, nPU, nSU, nch, nw, beta, alpha, DistAmp, noi_pw);

    // 同步并获取结果
    double *data = new double[nSU * nch * nw];
    cudaMemcpy(data, d_data, nSU * nch * nw * sizeof(double), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // 处理结果并返回
    std::vector<std::vector<std::vector<double>>> result(nSU, std::vector<std::vector<double>>(nch, std::vector<double>(nw)));
    for (int suIdx = 0; suIdx < nSU; ++suIdx)
    {
        for (int chIdx = 0; chIdx < nch; ++chIdx)
        {
            for (int sampleIdx = 0; sampleIdx < nw; ++sampleIdx)
            {
                result[suIdx][chIdx][sampleIdx] = data[suIdx * nch * nw + chIdx * nw + sampleIdx];
            }
        }
    }

    // 清理设备内存
    cudaFree(d_distances);
    cudaFree(d_data);
    cudaFree(d_noises);
    delete[] distances;
    delete[] noises;
    delete[] data;

    return result;
}
