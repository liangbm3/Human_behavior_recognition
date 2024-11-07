import numpy as np

# 读取 .npy 文件
data = np.load(r'/data2/songxinshuai/ICMEW2024-Track10/Model_inference/Mix_GCN/msg3d_joinmotiont3d_test_B.npy',allow_pickle=True)

print(data.shape)
# 将数组转换为可打印的字符串格式
np.set_printoptions(threshold=np.inf)  # 让所有数据都可以被打印出来
data_str = np.array2string(data, separator=',')

# 将内容保存到一个可复制的文本文件中
with open('/data2/songxinshuai/ICMEW2024-Track10/Model_inference/Mix_GCN/msg3d_joinmotiont3d_test_B.txt', 'w') as f:
    f.write(data_str)

print("Array content saved as a text file. You can now copy the content.")

