# src/python/data_loader_cuda.py

from torch.utils.data import Dataset
import data_generator_cuda_cpp
from utils.data_utils import (
    get_locations,
    calculate_distances,
    get_assignments
)
import torch
import os


class MyDatasetCUDA(Dataset):
    def __init__(self, train=True, save_data=True, save_dir='./generated_data'):
        # 获取 PU 和 SU 的位置
        locat_endpt, locat_centre = get_locations()
        # 计算距离字典
        dist_dict = calculate_distances(locat_endpt, locat_centre)
        # 获取频道分配和 SU 可观测的频道列表
        assign_dict, class_dir, nPU, description = get_assignments()

        # 定义数据生成器的参数
        DistAmp = 10.0
        SNR = -10.0
        alpha = 3.71
        beta = 10 ** 3.154
        nch = 20
        nw = 64
        dbsize_list = [100] * len(class_dir)  # 每个类的数据量，示例值

        # 指定 PSD 库目录
        psd_directory = '/home/coder/IRS_PartialObservation/src/python/utils/test_psd_lib'  # 提供实际的路径

        # 加载 PSDLibrary 对象
        psd_library = data_generator_cuda_cpp.load_PSD_library(psd_directory)

        # 调用 CUDA 版本的数据生成器
        print("Using CUDA version of data generator")
        self.data, self.labels = data_generator_cuda_cpp.generate_data(
            DistAmp,
            class_dir,  # class_dir 保持嵌套结构
            dbsize_list,
            nch,
            nw,
            {str(k): v for k, v in assign_dict.items()},  # 确保 assign_dict 的键为字符串
            SNR,
            {int(k): [float(x) for x in v] for k, v in dist_dict.items()},  # 确保 dist_dict 的键为整数，值为浮点数列表
            psd_library,  # 传递加载的 PSDLibrary 对象
            alpha,
            beta
        )

        # 数据预处理 - 将数据和标签转换为 PyTorch 张量
        self.data = [torch.tensor(sample, dtype=torch.float32) for sample in self.data]
        self.labels = [torch.tensor(label, dtype=torch.long) for label in self.labels]

        # 如果需要保存生成的数据
        if save_data:
            self.save_generated_data(save_dir, SNR)

    def save_generated_data(self, save_dir, SNR):
        """
        保存生成的数据为 .pth 文件
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # 准备保存的数据
        db = {
            'Description': f'Generated data with SNR {SNR} dB',
            'training data list': self.data,
            'training label list': self.labels,
            'time': 'TBD'  # 可以使用时间戳或其他方式记录生成时间
        }

        # 生成保存的文件名
        name = f'Data_SNR_{SNR}.pth'
        save_path = os.path.join(save_dir, name)

        # 保存数据
        torch.save(db, save_path)
        print(f'Dataset saved to: {save_path}')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# 测试代码
if __name__ == "__main__":
    # 使用 CUDA 版本
    dataset = MyDatasetCUDA(train=True, save_data=True, save_dir='/home/coder/generated_data')
