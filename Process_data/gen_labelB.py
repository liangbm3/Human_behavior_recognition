import numpy as np

# 生成一个长度为4599的全零数组
array = np.zeros(4307)

# 保存为 .npy 文件
np.save('./data/test_B_label.npy', array)
