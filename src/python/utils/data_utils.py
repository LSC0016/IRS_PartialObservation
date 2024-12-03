# src/python/utils/data_utils.py

import numpy as np
import matplotlib.pyplot as plt
import os

def get_locations():
    # 定义PU和SU的位置
    locat_endpt = {  # PU的位置
        0: [-2, 3**0.5],
        1: [0, 3**0.5],
        2: [2, 3**0.5],
        3: [-3, 0],
        4: [-1, 0],
        5: [1, 0],
        6: [3, 0],
        7: [-2, -3**0.5],
        8: [0, -3**0.5],
        9: [2, -3**0.5],
    }

    locat_centre = {  # SU的位置
        0: [-2, 3**0.5 / 3],
        1: [-1, 2/3 * 3**0.5],
        2: [0, 3**0.5 / 3],
        3: [1, 2/3 * 3**0.5],
        4: [2, 3**0.5 / 3],
        5: [-2, -3**0.5 / 3],
        6: [-1, -2/3 * 3**0.5],
        7: [0, -3**0.5 / 3],
        8: [1, -2/3 * 3**0.5],
        9: [2, -3**0.5 / 3],
    }
    return locat_endpt, locat_centre

def calculate_distances(locat_endpt, locat_centre):
    # 计算PU到SU的距离
    dist_dict = {
        i: [
            np.float64(np.linalg.norm(np.array(locat_endpt[i]) - np.array(locat_centre[j])))
            for j in range(len(locat_centre))
        ]
        for i in range(len(locat_endpt))
    }
    return dist_dict

def plot_locations(locat_endpt, locat_centre):
    # 绘制PU和SU的位置
    x_pu = [locat_endpt[i][0] for i in range(len(locat_endpt))]
    y_pu = [locat_endpt[i][1] for i in range(len(locat_endpt))]
    x_su = [locat_centre[i][0] for i in range(len(locat_centre))]
    y_su = [locat_centre[i][1] for i in range(len(locat_centre))]

    plt.figure(figsize=(5, 3))
    plt.scatter(x_pu, y_pu, label='PUs', marker='o')
    plt.scatter(x_su, y_su, label='SUs', marker='^')
    plt.legend()
    plt.xlabel('X-coordinate')
    plt.ylabel('Y-coordinate')
    plt.title('Locations of PUs and SUs')
    plt.grid(True)
    plt.show()

def get_assignments():
    # 定义频道分配和观测
    assign_dict10 = {
        'PU0': [0],
        'PU1': [1],
        'PU2': [2],
        'PU3': [3],
        'PU4': [4],
        'PU5': [5],
        'PU6': [6],
        'PU7': [7],
        'PU8': [8],
        'PU9': [9],
    }
    nPU = 10
    description10 = 'The bands allocated to each PU for 10-band case'

    class_dir10 = [
        [0, 3, 4],
        [0, 1, 4],
        [1, 4, 5],
        [1, 2, 5],
        [2, 5, 6],
        [3, 4, 7],
        [4, 7, 8],
        [4, 5, 8],
        [5, 8, 9],
        [5, 6, 9],
    ]

    assign_dict20 = {
        'PU0': [0],
        'PU1': [1, 10],
        'PU2': [2, 11, 14],
        'PU3': [3],
        'PU4': [4, 19],
        'PU5': [5, 13],
        'PU6': [6, 15, 17],
        'PU7': [7, 12, 18],
        'PU8': [8, 16],
        'PU9': [9],
    }
    description20 = 'The bands allocated to each PU for 20-band case, 3 single-band PUs, 4 double-band PUs, 3 triple-band PUs'

    class_dir20 = [[] for _ in range(len(class_dir10))]

    for SU in range(len(class_dir10)):
        for PU in class_dir10[SU]:
            class_dir20[SU].extend(assign_dict20['PU' + str(PU)])

    # 打印每个SU可观测的频道
    for SU in range(len(class_dir20)):
        print('SU', SU, 'observable bands:', class_dir20[SU])

    return assign_dict20, class_dir20, nPU, description20

def save_PSD_lib(PSD_lib, directory):
    """
    将 PSD_lib 保存为 CSV 文件。

    参数：
    - PSD_lib：PSDLibrary 对象，包含 PSD 数据，格式为 {PU编号: [样本列表]}。
    - directory：保存 CSV 文件的目录。
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    for PU in PSD_lib.data:
        # 根据 C++ 代码的期望调整 PU 编号
        PU_index = PU  # 假设 PU 编号从 1 开始
        filename = os.path.join(directory, f"PSD_PU{PU_index}.csv")
        with open(filename, 'w') as f:
            for sample in PSD_lib.data[PU]:
                line = ','.join(map(str, sample))
                f.write(line + '\n')

# 添加测试代码
if __name__ == "__main__":
    # 测试 get_locations 函数
    locat_endpt, locat_centre = get_locations()

    # 测试 calculate_distances 函数
    dist_dict = calculate_distances(locat_endpt, locat_centre)

    # 打印距离字典
    for PU in range(len(dist_dict)):
        distances = dist_dict[PU]
        print(f"PU {PU} :  {distances}")

    # 测试保存 PSD_lib（示例）
    class PSDLibrary:
        def __init__(self):
            self.description = "Test PSD library"
            self.data = {
                1: [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
                2: [[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]],
                3: [[1.3, 1.4, 1.5], [1.6, 1.7, 1.8]],
                4: [[1.9, 2.0, 2.1], [2.2, 2.3, 2.4]],
                5: [[2.5, 2.6, 2.7], [2.8, 2.9, 3.0]],
                6: [[3.1, 3.2, 3.3], [3.4, 3.5, 3.6]],
            }

    PSD_lib = PSDLibrary()
    save_directory = 'test_psd_lib'
    save_PSD_lib(PSD_lib, save_directory)
    print(f"\nPSD library saved to '{save_directory}' directory.")
