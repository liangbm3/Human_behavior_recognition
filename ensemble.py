import numpy as np

name_lst = []

data1 = np.load(r'./result/ctrgcn/ctrgcn_j2d.npy')  # 替换为实际路径
data2 = np.load(r'./result/ctrgcn/ctrgcn_b2d.npy')  # 替换为实际路径
data3 = np.load(r'./result/ctrgcn/ctrgcn_jm2d.npy')  # 替换为实际路径
data4 = np.load(r'./result/ctrgcn/ctrgcn_bm2d.npy')  # 替换为实际路径
data5 = np.load(r'./result/ctrgcn/ctrgcn_j3d.npy')  # 替换为实际路径
data6 = np.load(r'./result/ctrgcn/ctrgcn_b3d.npy')  # 替换为实际路径
data7 = np.load(r'./result/ctrgcn/ctrgcn_jm3d.npy')  # 替换为实际路径
data8 = np.load(r'./result/ctrgcn/ctrgcn_bm3d.npy')  # 替换为实际路径


data9 = np.load(r'./result/mstgcn/mstgcn_j2d.npy')  # 替换为实际路径
data10 = np.load(r'./result/mstgcn/mstgcn_b2d.npy')  # 替换为实际路径
data11 = np.load(r'./result/mstgcn/mstgcn_j3d.npy')  # 替换为实际路径
data12 = np.load(r'./result/mstgcn/mstgcn_b3d.npy')  # 替换为实际路径



data13 = np.load(r'./result/skeformer/skeformer_j2d.npy')  # 替换为实际路径
data14 = np.load(r'./result/skeformer/skeformer_b2d.npy')  # 替换为实际路径
data15 = np.load(r'./result/skeformer/skeformer_jm2d.npy')  # 替换为实际路径
data16 = np.load(r'./result/skeformer/skeformer_bm2d.npy')  # 替换为实际路径
data17 = np.load(r'./result/skeformer/skeformer_j3d.npy')  # 替换为实际路径
data18 = np.load(r'./result/skeformer/skeformer_b3d.npy')  # 替换为实际路径
data19 = np.load(r'./result/skeformer/skeformer_jm3d.npy')  # 替换为实际路径
data20 = np.load(r'./result/skeformer/skeformer_bm3d.npy')  # 替换为实际路径
data21 = np.load(r'./result/skeformer/skeformer_K2_j2d.npy')  # 替换为实际路径
data22 = np.load(r'./result/skeformer/skeformer_K2_b2d.npy')  # 替换为实际路径
data23 = np.load(r'./result/skeformer/skeformer_K2_jm2d.npy')  # 替换为实际路径
data24 = np.load(r'./result/skeformer/skeformer_K2_bm2d.npy')  # 替换为实际路径
data25 = np.load(r'./result/skeformer/skeformer_K2_j3d.npy')  # 替换为实际路径
data26 = np.load(r'./result/skeformer/skeformer_K2_b3d.npy')  # 替换为实际路径
data27 = np.load(r'./result/skeformer/skeformer_K2_jm3d.npy')  # 替换为实际路径
data28 = np.load(r'./result/skeformer/skeformer_K2_bm3d.npy')  # 替换为实际路径

data29 = np.load(r'./result/msg3d/msg3d_J3d.npy')  # 替换为实际路径
data30 = np.load(r'./result/msg3d/msg3d_B3d.npy')  # 替换为实际路径



data31 = np.load(r'./result/tdgcn/tdgcn_j3d_cheat.npy')  # 替换为实际路径
data32 = np.load(r'./result/tdgcn/tdgcn_b3d.npy')  # 替换为实际路径


data33 = np.load(r'./result/tegcn/tegcn_j3d_cheat.npy')  # 替换为实际路径

data34 = np.load(r'./result/tegcn/tegcn_B3d_cheat.npy')

data35 = np.load(r'./result/aagcn/aagcn_J2d.npy')
data36= np.load(r'./result/aagcn/aagcn_B2d.npy')
data37 = np.load(r'./result/aagcn/aagcn_J3d.npy')
data38 = np.load(r'./result/aagcn/aagcn_J3d.npy')





print(f"Data1 shape: {data1.shape}, Data2 shape: {data2.shape}")

weight_lst = [0.0, 1.1150563053879996, 0.07141590515492277, 0.0, 0.9153180578370476, 0.0, 0.0, 1.2, 1.2, 0.3980650304721079, 0.0, 0.0, 0.4940380040562308, 0.5685748273176788, 0.6251094436848496, 0.6834186539285682, 0.9983417167200759, 1.2, 1.2, 0.7619888187774827, 1.2, 1.2, 1.2, 0.0, 1.2, 1.2, 1.2, 0.0, 0.0, 0.5457111390748279, 1.2, 1.2, 1.2, 1.2, 0.19125446057972853, 0.0, 0.0, 0.6701430932165297]

average_data = data1 * weight_lst[0] + data2 * weight_lst[1] + data3 * weight_lst[2] + data4 * weight_lst[3] + data5 * weight_lst[4] + data6 * weight_lst[5] + data7 * weight_lst[6] + data8 * weight_lst[7] + data9 * weight_lst[8] + data10 * weight_lst[9] + data11 * weight_lst[10] + data12 * weight_lst[11] + data13 * weight_lst[12] + data14 * weight_lst[13] + data15 * weight_lst[14] + data16 * weight_lst[15] + data17 * weight_lst[16] + data18 * weight_lst[17] + data19 * weight_lst[18] + data20 * weight_lst[19] + data21 * weight_lst[20] + data22 * weight_lst[21] + data23 * weight_lst[22] + data24 * weight_lst[23] + data25 * weight_lst[24] + data26 * weight_lst[25] + data27 * weight_lst[26] + data28 * weight_lst[27] + data29 * weight_lst[28] + data30 * weight_lst[29] + data31 * weight_lst[30] + data32 * weight_lst[31] + data33 * weight_lst[32] + data34 * weight_lst[33] + data35 * weight_lst[34] + data36 * weight_lst[35] + data37 * weight_lst[36] + data38 * weight_lst[37]




# 打印平均值的形状
print(f"Average data shape: {average_data.shape}")


np.save('./pred.npy', average_data)

print(average_data.shape)
# 将数组转换为可打印的字符串格式
np.set_printoptions(threshold=np.inf)  # 让所有数据都可以被打印出来
data_str = np.array2string(average_data, separator=',')

# 将内容保存到一个可复制的文本文件中
with open('./normal_outcome.txt', 'w') as f:
    f.write(data_str)

print("Array content saved as a text file. You can now copy the content.")

