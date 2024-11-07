import numpy as np

name_lst = ['ctrgcn_b3d.txt','ctrgcn_bm3d.txt','ctrgcn_j2d.txt','ctrgcn_j3d.txt','ctrgcn_jm3d.txt',
            'msg3d_b3d.txt','msg3d_j3d.txt','msg3d_jm3d.txt',
            'mstgcn_b3d.txt','mstgcn_j2d.txt','mstgcn_j3d.txt',
            'skeformer_bone3d.txt','skeformer_joint3d.txt','skeformer_K2_bone3d.txt','skeformer_K2_joint3d.txt','skeformer_V2_jointmotion3d.txt','skeformer_V2_bonemotion_3d.txt'
            ,'tdgcn_b3d.txt','tdgcn_j2d.txt','tdgcn_j3d.txt',
            'tegcn_b3d.txt','tegcn_j3d.txt',
            'aagcn_j3d.txt','aagcn_b3d.txt']



# 导入两个 .npy 文件
data1 = np.load(r'./ensemble_npy/aagcn_B_3d_result.npy')  # 替换为实际路径
data2 = np.load(r'./ensemble_npy/aagcn_BM_3d_result.npy')  # 替换为实际路径
data3 = np.load(r'./ensemble_npy/aagcn_J_2d_result.npy')  # 替换为实际路径
data4 = np.load(r'./ensemble_npy/aagcn_J_3d_result.npy')  # 替换为实际路径
data5 = np.load(r'./ensemble_npy/aagcn_JM_3d_result.npy')  # 替换为实际路径
data6 = np.load(r'./ensemble_npy/ctrgcn_B_3d_result.npy')  # 替换为实际路径
data7 = np.load(r'./ensemble_npy/ctrgcn_BM_3d_result.npy')  # 替换为实际路径
data8 = np.load(r'./ensemble_npy/ctrgcn_J_2d_result.npy')  # 替换为实际路径
data9 = np.load(r'./ensemble_npy/ctrgcn_J_3d_result.npy')  # 替换为实际路径
data10 = np.load(r'./ensemble_npy/ctrgcn_JM_3d_result.npy')  # 替换为实际路径
data11 = np.load(r'./ensemble_npy/mixformer_B_result.npy')  # 替换为实际路径
data12 = np.load(r'./ensemble_npy/mixformer_BM_result.npy')  # 替换为实际路径
data13 = np.load(r'./ensemble_npy/mixformer_J_result.npy')  # 替换为实际路径
data14 = np.load(r'./ensemble_npy/mixformer_JM_result.npy')  # 替换为实际路径
data15 = np.load(r'./ensemble_npy/mixformer_k2_B3d.npy')  # 替换为实际路径
data16 = np.load(r'./ensemble_npy/mixformer_k2_J3d.npy')  # 替换为实际路径
data17 = np.load(r'./ensemble_npy/msg3d_B_3d_result.npy')  # 替换为实际路径
data18 = np.load(r'./ensemble_npy/msg3d_J_3d_result.npy')  # 替换为实际路径
data19 = np.load(r'./ensemble_npy/msg3d_JM_3d_result.npy')  # 替换为实际路径
data20 = np.load(r'./ensemble_npy/mstgcn_B_3d_result.npy')  # 替换为实际路径
data21 = np.load(r'./ensemble_npy/mstgcn_J_2d_result.npy')  # 替换为实际路径
data22 = np.load(r'./ensemble_npy/mstgcn_J_3d_result.npy')  # 替换为实际路径
data23 = np.load(r'./ensemble_npy/tdgcn_B_3d_result.npy')  # 替换为实际路径
data24 = np.load(r'./ensemble_npy/tdgcn_J_2d_result.npy')  # 替换为实际路径
data25 = np.load(r'./ensemble_npy/tdgcn_J_3d_result.npy')  # 替换为实际路径
data26 = np.load(r'./ensemble_npy/tegcn_B_3d_result.npy')  # 替换为实际路径
data27 = np.load(r'./ensemble_npy/tegcn_J_3d_result.npy')  # 替换为实际路径


print(f"Data1 shape: {data1.shape}, Data2 shape: {data2.shape}")

weight_lst = [10.0, -0.5, -0.5, 10.0, 4.993956733981186, 4.336927387389084, -0.5, 5.990988608077026, -0.5, 4.137306018513151, 3.7154011970028957, 0.025253145953622047, 4.054641265199987, 5.8246462160838774, 3.3924801532936555, 7.855742986734876, 10.0, -0.5, 9.806713326244559, 8.513403779270622, 10.0, 1.5466514521653623, 10.0, 7.472887204844962]

average_data = data1 * weight_lst[0] + data2 * weight_lst[1] + data3 * weight_lst[2] + data4 * weight_lst[3] + data5 * weight_lst[4] + data6 * weight_lst[5] + data7 * weight_lst[6] + data8 * weight_lst[7] + data9 * weight_lst[8] + data10 * weight_lst[9] + data11 * weight_lst[10] + data12 * weight_lst[11] + data13 * weight_lst[12] + data14 * weight_lst[13] + data15 * weight_lst[14] + data16 * weight_lst[15] + data17 * weight_lst[16] + data18 * weight_lst[17] + data19 * weight_lst[18] + data20 * weight_lst[19]


# 打印平均值的形状
print(f"Average data shape: {average_data.shape}")

np.save(r"./pred.npy", average_data)

