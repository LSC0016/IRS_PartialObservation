# main.py

from data_loader import MyDataset
from utils.data_utils import get_locations, plot_locations
import torch

if __name__ == "__main__":
    # 可选：绘制PU和SU的位置
    locat_endpt, locat_centre = get_locations()
    plot_locations(locat_endpt, locat_centre)
    
    # 创建数据集实例
    dataset = MyDataset(train=True)
    
    # 查看数据集的大小
    print(f"Total number of samples: {len(dataset)}")
    
    # 获取第一个样本
    data_sample, label_sample = dataset[0]
    print("Data sample shape:", data_sample.shape)
    print("Label sample:", label_sample.shape)
    
    # 打印数据和标签的内容
    print("Data sample:", data_sample)
    print("Label sample:", label_sample)
    
    # 遍历前几个样本
    for idx in range(5):
        data, label = dataset[idx]
        print(f"\nSample {idx}:")
        print("Data shape:", data.shape)
        print("Label shape:", label.shape)
        print("Data:", data)
        print("Label:", label)
    
    # 如果需要，可以创建数据加载器
    from torch.utils.data import DataLoader
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # 迭代数据加载器中的批次
    for batch_idx, (data_batch, label_batch) in enumerate(data_loader):
        print(f"\nBatch {batch_idx}:")
        print("Data batch shape:", data_batch.shape)
        print("Label batch shape:", label_batch.shape)
        if batch_idx >= 1:  # 仅演示前两个批次
            break

