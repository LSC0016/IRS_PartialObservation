import torch
import os
from torch.utils.data import Dataset
from utils.data_utils import (
    get_locations,
    calculate_distances,
    get_assignments
)


class MyDataset(Dataset):
    def __init__(self, train=True, save_data=True, save_dir='./generated_data', use_cuda=False):
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

        # 选择使用 CUDA 版本还是 OpenMP 版本的数据生成器
        if use_cuda:
            print("Using CUDA version of data generator")
            import data_generator_cuda_cpp  # 在这里动态导入 CUDA 版本
            self.data, self.labels = data_generator_cuda_cpp.generate_data(
                DistAmp,
                class_dir,
                dbsize_list,
                nch,
                nw,
                assign_dict,
                SNR,
                dist_dict,
                psd_directory,  # 传递 PSD 库的路径，由 C++ 端加载
                alpha,
                beta
            )
        else:
            print("Using OpenMP version of data generator")
            import data_generator_openmp_cpp  # 在这里动态导入 OpenMP 版本
            self.data, self.labels = data_generator_openmp_cpp.generate_data(
                DistAmp,
                class_dir,
                dbsize_list,
                nch,
                nw,
                assign_dict,
                SNR,
                dist_dict,
                psd_directory,  # 传递 PSD 库的路径，由 C++ 端加载
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
    # 可以通过更改 `use_cuda` 参数来选择使用 CUDA 或 OpenMP 版本
    dataset = MyDataset(train=True, save_data=True, save_dir='/home/coder/generated_data', use_cuda=True)
