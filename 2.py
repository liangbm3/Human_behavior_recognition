import os
import numpy as np

# 指定文件夹路径
folder_path = '/data/coding/MixFormerGcn/Process_data/data'  # 请替换为你的文件夹路径

# 获取文件夹中的所有文件
files = os.listdir(folder_path)

# 遍历文件夹中的所有文件
for file in files:
    # 只处理 .npy 文件
    if file.endswith('.npy'):
        file_path = os.path.join(folder_path, file)
        try:
            # 加载 .npy 文件
            data = np.load(file_path)
            # 打印文件名及其形状
            print(f'{file}: {data.shape}')
        except Exception as e:
            print(f'无法加载 {file}: {e}')
