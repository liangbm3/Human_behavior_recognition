import numpy as np
import os

folder_path = '/data/coding/MixFormerGcn/Process_data/data'
# 生成一个长度为4599的全零数组
data=np.load(os.path.join(folder_path, 'test_joint.npy'))
array = np.zeros(data.shape[0])

# 保存为 .npy 文件
np.save('./data/test_label.npy', array)
print('保存成功')
