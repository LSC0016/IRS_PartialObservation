# src/python/data_loader.py

from torch.utils.data import Dataset
import data_generator_cpp  # 导入C++数据生成器模块
from utils.data_utils import (
    get_locations,
    calculate_distances,
    get_assignments,
    plot_locations  # 如果需要可视化
)
import numpy as np
import torch


class MyDataset(Dataset):
    def __init__(self, train=True):
        # 获取PU和SU的位置
        locat_endpt, locat_centre = get_locations()
        # 计算距离字典
        dist_dict = calculate_distances(locat_endpt, locat_centre)
        # 获取频道分配和SU可观测的频道列表
        assign_dict, class_dir, nPU, description = get_assignments()
    
        # 定义数据生成器的参数
        self.DistAmp = 10.0
        self.class_dir = class_dir
        self.dbsize_list = [100] * len(class_dir)  # 每个类的数据量，示例值
        self.nch = 20
        self.nw = 64
        self.assign_dict = assign_dict
        self.SNR = -10.0
        self.dist_dict = dist_dict
        self.alpha = 3.71
        self.beta = np.power(10, 3.154)
        self.nPU = nPU
        self.nSU = len(class_dir)
    
        # 加载PSD库
        psd_directory = 'path_to_psd_files'  # 您需要提供实际的路径
        self.PSD_lib = data_generator_cpp.load_PSD_library(psd_directory)
    
        # 调用C++数据生成器
        self.data, self.labels = data_generator_cpp.generate_data(
            self.DistAmp,
            self.class_dir,
            self.dbsize_list,
            self.nch,
            self.nw,
            self.assign_dict,
            self.SNR,
            self.dist_dict,
            self.PSD_lib,
            self.alpha,
            self.beta
        )
    
        # 数据预处理
        # 将数据和标签转换为PyTorch张量
        self.data = [torch.tensor(sample, dtype=torch.float32) for sample in self.data]
        self.labels = [torch.tensor(label, dtype=torch.long) for label in self.labels]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
