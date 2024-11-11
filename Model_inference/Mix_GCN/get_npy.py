import numpy as np
import os

# 指定文件夹路径
folder_path = r'./Model_inference/Mix_GCN/ensemble_npy/'

# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    if filename.endswith('.npy'):
        file_path = os.path.join(folder_path, filename)
        
        # 读取 .npy 文件
        data = np.load(file_path, allow_pickle=True)
        
        # 打印数组的形状
        print(f"{filename} shape: {data.shape}")
        
        # 将数组转换为可打印的字符串格式
        np.set_printoptions(threshold=np.inf)  # 让所有数据都可以被打印出来
        data_str = np.array2string(data, separator=',')
        
        # 保存为同名的 .txt 文件
        txt_filename = filename.replace('.npy', '.txt')
        txt_file_path = os.path.join(folder_path, txt_filename)
        with open(txt_file_path, 'w') as f:
            f.write(data_str)
        
        print(f"{filename} content saved as {txt_filename}.")
