import numpy as np

# 加载四个 .npy 文件
joint = np.load('/data2/songxinshuai/ICMEW2024-Track10/Process_data/data/train_joint.npy')  # 形状: (N, C, T, V, M)
bone = np.load('/data2/songxinshuai/ICMEW2024-Track10/Process_data/data/train_bone.npy')    # 形状: (N, C, T, V, M)
jm = np.load('/data2/songxinshuai/ICMEW2024-Track10/Process_data/data/train_joint_motion.npy')        # 形状: (N, C, T, V, M)
bm = np.load('/data2/songxinshuai/ICMEW2024-Track10/Process_data/data/train_bone_motion.npy')        # 形状: (N, C, T, V, M)

# 检查四个文件的形状是否一致
assert joint.shape == bone.shape == jm.shape == bm.shape, "形状不一致！"

# 拼接四个模态的 C 维度
# 将它们沿着第 1 维（C 维）拼接，C 的总数应该是 4 * 原始的 C
combined = np.concatenate((joint, bone, jm, bm), axis=1)  # 沿着 C 维度拼接

# 检查拼接后的形状
print("拼接后的形状：", combined.shape)  # 形状应该是 (N, 4*C, T, V, M)

# 保存拼接后的结果为新的 .npy 文件
np.save('/data2/songxinshuai/ICMEW2024-Track10/Process_data/combined_train.npy', combined)

print("拼接完成，结果保存在 'combined_data.npy")



# 加载四个 .npy 文件
joint = np.load('/data2/songxinshuai/ICMEW2024-Track10/Process_data/data/val_joint.npy')  # 形状: (N, C, T, V, M)
bone = np.load('/data2/songxinshuai/ICMEW2024-Track10/Process_data/data/val_bone.npy')    # 形状: (N, C, T, V, M)
jm = np.load('/data2/songxinshuai/ICMEW2024-Track10/Process_data/data/val_joint_motion.npy')        # 形状: (N, C, T, V, M)
bm = np.load('/data2/songxinshuai/ICMEW2024-Track10/Process_data/data/val_bone_motion.npy')        # 形状: (N, C, T, V, M)

# 检查四个文件的形状是否一致
assert joint.shape == bone.shape == jm.shape == bm.shape, "形状不一致！"

# 拼接四个模态的 C 维度
# 将它们沿着第 1 维（C 维）拼接，C 的总数应该是 4 * 原始的 C
combined = np.concatenate((joint, bone, jm, bm), axis=1)  # 沿着 C 维度拼接

# 检查拼接后的形状
print("拼接后的形状：", combined.shape)  # 形状应该是 (N, 4*C, T, V, M)

# 保存拼接后的结果为新的 .npy 文件
np.save('/data2/songxinshuai/ICMEW2024-Track10/Process_data/combined_val.npy', combined)

print("拼接完成，结果保存在 'combined_data.npy")




# 加载四个 .npy 文件
joint = np.load('/data2/songxinshuai/ICMEW2024-Track10/Process_data/data/test_joint.npy')  # 形状: (N, C, T, V, M)
bone = np.load('/data2/songxinshuai/ICMEW2024-Track10/Process_data/data/test_bone.npy')    # 形状: (N, C, T, V, M)
jm = np.load('/data2/songxinshuai/ICMEW2024-Track10/Process_data/data/test_joint_motion.npy')        # 形状: (N, C, T, V, M)
bm = np.load('/data2/songxinshuai/ICMEW2024-Track10/Process_data/data/test_bone_motion.npy')        # 形状: (N, C, T, V, M)

# 检查四个文件的形状是否一致
assert joint.shape == bone.shape == jm.shape == bm.shape, "形状不一致！"

# 拼接四个模态的 C 维度
# 将它们沿着第 1 维（C 维）拼接，C 的总数应该是 4 * 原始的 C
combined = np.concatenate((joint, bone, jm, bm), axis=1)  # 沿着 C 维度拼接

# 检查拼接后的形状
print("拼接后的形状：", combined.shape)  # 形状应该是 (N, 4*C, T, V, M)

# 保存拼接后的结果为新的 .npy 文件
np.save('/data2/songxinshuai/ICMEW2024-Track10/Process_data/combined_test.npy', combined)

print("拼接完成，结果保存在 'combined_data.npy")